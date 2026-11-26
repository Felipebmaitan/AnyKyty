#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <initializer_list>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

size_t BlockAt(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

bool LeaveAfter(Block& exit, Block& stay, Value done, uint32_t limit, bool mark) {
	auto& term = exit.terminator;
	if (exit.condition.IsEmpty() || term.kind != CFG::TerminatorKind::ConditionalBranch) return false;
	const bool stay_on_true = term.true_block == &stay;
	if (!stay_on_true && term.false_block != &stay) return false;
	const auto add = [&](ValueOpcode op, std::initializer_list<Value> args) {
		return Value(&*exit.PrependNewInst(exit.end(), op, args));
	};
	const auto under = add(ValueOpcode::ULessThan32, {done, Value(limit)});
	Value      over;
	if (mark || !stay_on_true) over = add(ValueOpcode::LogicalNot, {under});
	if (mark) {
		const auto stays = stay_on_true ? exit.condition : add(ValueOpcode::LogicalNot, {exit.condition});
		add(ValueOpcode::LoopGuardMark, {add(ValueOpcode::LogicalAnd, {stays, over})});
	}
	exit.condition = stay_on_true ? add(ValueOpcode::LogicalAnd, {exit.condition, under})
	                              : add(ValueOpcode::LogicalOr, {exit.condition, over});
	return true;
}

bool ZeroAvailable(Type type) {
	switch (type) {
		case Type::U1:
		case Type::U8:
		case Type::U16:
		case Type::U32:
		case Type::U64:
		case Type::F16:
		case Type::F32:
		case Type::F64:
		case Type::U32x2:
		case Type::U32x3:
		case Type::U32x4:
		case Type::F32x2: return true;
		default: return false;
	}
}

Value ZeroOf(Block& block, Type type) {
	const auto at = [&](ValueOpcode op, std::initializer_list<Value> args) {
		return Value(&*block.PrependNewInst(block.end(), op, args));
	};
	switch (type) {
		case Type::U1: return Value(false);
		case Type::U8: return Value(uint8_t {0});
		case Type::U16: return Value(uint16_t {0});
		case Type::U32: return Value(0u);
		case Type::U64: return Value(uint64_t {0});
		case Type::F16: return at(ValueOpcode::BitCastF16U16, {Value(uint16_t {0})});
		case Type::F32: return at(ValueOpcode::BitCastF32U32, {Value(0u)});
		case Type::F64: return at(ValueOpcode::BitCastF64U64, {Value(uint64_t {0})});
		case Type::U32x2: return at(ValueOpcode::CompositeConstructU32x2, {Value(0u), Value(0u)});
		case Type::U32x3: return at(ValueOpcode::CompositeConstructU32x3, {Value(0u), Value(0u), Value(0u)});
		case Type::U32x4: return at(ValueOpcode::CompositeConstructU32x4, {Value(0u), Value(0u), Value(0u), Value(0u)});
		case Type::F32x2: {
			const auto zero = at(ValueOpcode::BitCastF32U32, {Value(0u)});
			return at(ValueOpcode::CompositeConstructF32x2, {zero, zero});
		}
		default: return {};
	}
}

Block* EveryIterationExit(Program& program, Block& header, Block* latch, size_t first, size_t last, Block*& stay) {
	auto* merge = header.terminator.merge_block;
	auto* at    = &header;
	for (size_t steps = 0; at != nullptr && at != latch && steps <= last - first; ++steps) {
		const auto index = BlockAt(program, at);
		if (index < first || index > last) return nullptr;
		const auto& term = at->terminator;
		if (term.kind == CFG::TerminatorKind::Branch) {
			at = at != &header && term.loop_header && term.merge_block != nullptr ? term.merge_block : term.true_block;
			continue;
		}
		if (term.kind != CFG::TerminatorKind::ConditionalBranch || at->condition.IsEmpty()) return nullptr;
		if (term.true_block != merge && term.false_block != merge) {
			if (term.merge_block == nullptr || (!term.loop_header && BlockAt(program, term.merge_block) <= index)) return nullptr;
			at = term.merge_block;
			continue;
		}
		stay = term.true_block == merge ? term.false_block : term.true_block;
		return stay != merge ? at : nullptr;
	}
	return nullptr;
}

bool InsideLoop(const Program& program, const Block* block, size_t first, size_t last) {
	const auto at = BlockAt(program, block);
	return at >= first && at <= last;
}

bool BodyValuesLeave(const Program& program, size_t first, size_t last, const Block* merge) {
	for (auto at = first + 1; at <= last; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			for (const auto& use: inst.Uses()) {
				const auto* block = use.user->Parent();
				if (InsideLoop(program, block, first, last)) continue;
				if (block == merge && use.user->GetOpcode() == ValueOpcode::Phi) continue;
				return true;
			}
		}
	}
	for (size_t at = 0; at < program.blocks.size(); ++at) {
		if (at >= first && at <= last) continue;
		for (const auto value: {program.blocks[at]->condition, program.blocks[at]->indirect_target}) {
			const auto* inst = value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
			if (inst != nullptr && inst->Parent() != nullptr && InsideLoop(program, inst->Parent(), first + 1, last)) return true;
		}
	}
	return false;
}

bool DominatesLatch(const Program& program, const Block* block, const Block& header, const Block* latch, size_t first,
                    size_t last) {
	if (block == &header || block == latch) return true;
	std::vector<bool>         seen(program.blocks.size());
	std::vector<const Block*> stack {&header};
	seen[first] = true;
	while (!stack.empty()) {
		const auto* at = stack.back();
		stack.pop_back();
		for (const auto* next: at->ImmSuccessors()) {
			if (next == block || !InsideLoop(program, next, first, last)) continue;
			if (next == latch) return false;
			const auto index = BlockAt(program, next);
			if (seen[index]) continue;
			seen[index] = true;
			stack.push_back(next);
		}
	}
	return true;
}

Value ExitValue(const Program& program, const Inst& phi, const Block& header, const Block* latch, size_t first, size_t last) {
	Value common;
	for (size_t index = 0; index < phi.NumArgs(); ++index) {
		if (!InsideLoop(program, phi.PhiBlock(index), first, last)) return {};
		const auto value = phi.Arg(index).Resolve();
		if (index != 0 && !(value == common)) return {};
		common = value;
	}
	if (common.IsEmpty() || common.IsImmediate()) return common;
	const auto* inst = common.TryInstruction();
	if (inst == nullptr || inst->Parent() == nullptr) return {};
	if (!InsideLoop(program, inst->Parent(), first, last)) return common;
	return DominatesLatch(program, inst->Parent(), header, latch, first, last) ? common : Value {};
}

bool HasBarrier(const Block& block) {
	return std::ranges::any_of(block, [](const Inst& inst) { return inst.GetOpcode() == ValueOpcode::Barrier; });
}

bool ReachesBarrier(const Program& program, const Block& header) {
	std::vector<bool>         seen(program.blocks.size());
	std::vector<const Block*> stack {&header};
	seen[BlockAt(program, &header)] = true;
	while (!stack.empty()) {
		const auto* at = stack.back();
		stack.pop_back();
		if (HasBarrier(*at)) return true;
		for (const auto* next: at->ImmSuccessors()) {
			const auto index = BlockAt(program, next);
			if (index >= seen.size() || seen[index]) continue;
			seen[index] = true;
			stack.push_back(next);
		}
	}
	return false;
}

}

const char* GuardLoop(Program& program, Block& header, uint32_t limit, LoopExit mode) {
	const auto first = BlockAt(program, &header);
	const auto last  = BlockAt(program, header.terminator.continue_block);
	auto*      merge = header.terminator.merge_block;
	Block*     latch = nullptr;
	for (auto* pred: header.ImmPredecessors()) {
		const auto at = BlockAt(program, pred);
		if (at < first || at > last) continue;
		if (latch != nullptr) return "several back edges";
		latch = pred;
	}
	if (latch == nullptr || merge == nullptr || merge == &header) return "no back edge or no merge block";
	auto&      term        = latch->terminator;
	const bool conditional = term.kind == CFG::TerminatorKind::ConditionalBranch && !latch->condition.IsEmpty() &&
	                         (term.true_block == &header || term.false_block == &header);
	const bool plain       = term.kind == CFG::TerminatorKind::Branch && term.true_block == &header &&
	                         std::ranges::find(latch->ImmSuccessors(), merge) == latch->ImmSuccessors().end();
	if (!conditional && !plain) return "back edge is neither conditional nor plain";
	const bool mark      = mode == LoopExit::Guard;
	const auto zero_exit = [&](const Inst& phi) { return mode == LoopExit::EmptyWalk || !phi.HasUses(); };
	Block*     exit      = nullptr;
	Block*     stay      = nullptr;
	if (plain) {
		exit = EveryIterationExit(program, header, latch, first, last, stay);
		if (exit == nullptr) {
			if (BodyValuesLeave(program, first, last, merge)) return "a value of the loop body is used after the loop";
			for (const auto& inst: *merge) {
				if (inst.GetOpcode() != ValueOpcode::Phi) break;
				if (zero_exit(inst) ? !ZeroAvailable(inst.GetType())
				                    : ExitValue(program, inst, header, latch, first, last).IsEmpty()) {
					return "a merge value has no known value on a new exit edge";
				}
			}
		}
	}
	auto& count = *header.PrependNewInst(header.begin(), ValueOpcode::Phi);
	count.SetFlags(Type::U32);
	const auto body = std::ranges::find_if(header, [](const Inst& inst) { return inst.GetOpcode() != ValueOpcode::Phi; });
	auto&      next = *header.PrependNewInst(body, ValueOpcode::IAdd32, {Value(&count), Value(1u)});
	for (auto* pred: header.ImmPredecessors()) {
		count.AddPhiOperand(pred, pred == latch ? Value(&next) : Value(0u));
	}
	if (conditional) {
		LeaveAfter(*latch, header, Value(&next), limit, mark);
		return nullptr;
	}
	if (exit != nullptr) {
		LeaveAfter(*exit, *stay, Value(&count), limit, mark);
		return nullptr;
	}
	for (auto& inst: *merge) {
		if (inst.GetOpcode() != ValueOpcode::Phi) break;
		inst.AddPhiOperand(latch, zero_exit(inst) ? ZeroOf(*latch, inst.GetType())
		                                          : ExitValue(program, inst, header, latch, first, last));
	}
	auto& under = *latch->PrependNewInst(latch->end(), ValueOpcode::ULessThan32, {Value(&next), Value(limit)});
	if (mark) {
		latch->PrependNewInst(latch->end(), ValueOpcode::LoopGuardMark,
		                      {Value(&*latch->PrependNewInst(latch->end(), ValueOpcode::LogicalNot, {Value(&under)}))});
	}
	latch->condition = Value(&under);
	term.kind        = CFG::TerminatorKind::ConditionalBranch;
	term.true_block  = &header;
	term.false_block = merge;
	latch->AddBranch(merge);
	return nullptr;
}

LoopGuards GuardComputeLoops(Program& program, uint32_t limit, bool several_waves, std::span<Block* const> bounded) {
	LoopGuards guards;
	if (program.dispatcher_fallback) {
		guards.unguarded = 1;
		return guards;
	}
	const bool rendezvous =
	    several_waves && std::ranges::any_of(program.blocks, [](const Block* block) { return HasBarrier(*block); });
	std::vector<Block*> headers;
	for (auto* block: program.blocks) {
		const auto& term = block->terminator;
		if (term.loop_header && term.continue_block != nullptr && term.merge_block != nullptr) headers.push_back(block);
	}
	for (auto* header: headers) {
		if (std::ranges::find(bounded, header) != bounded.end()) {
			++guards.bounded;
			continue;
		}
		const char* refusal = rendezvous && ReachesBarrier(program, *header)
		                          ? "a barrier shared by several waves can follow the loop exit"
		                          : GuardLoop(program, *header, limit, LoopExit::Guard);
		if (refusal == nullptr) {
			++guards.guarded;
			continue;
		}
		++guards.unguarded;
		guards.refusals.emplace_back(header->id, refusal);
	}
	return guards;
}

}

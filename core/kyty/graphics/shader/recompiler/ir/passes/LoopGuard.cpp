#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <initializer_list>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

size_t BlockAt(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

bool LeaveAfter(Block& exit, Block& stay, Value next, uint32_t limit, bool mark) {
	auto& term = exit.terminator;
	if (exit.condition.IsEmpty() || term.kind != CFG::TerminatorKind::ConditionalBranch) return false;
	const bool stay_on_true = term.true_block == &stay;
	if (!stay_on_true && term.false_block != &stay) return false;
	auto& under = *exit.PrependNewInst(exit.end(), ValueOpcode::ULessThan32, {next, Value(limit)});
	if (mark) {
		exit.PrependNewInst(exit.end(), ValueOpcode::LoopGuardMark,
		                    {Value(&*exit.PrependNewInst(exit.end(), ValueOpcode::LogicalNot, {Value(&under)}))});
	}
	if (stay_on_true) {
		exit.condition = Value(&*exit.PrependNewInst(exit.end(), ValueOpcode::LogicalAnd, {exit.condition, Value(&under)}));
	} else {
		auto& over     = *exit.PrependNewInst(exit.end(), ValueOpcode::LogicalNot, {Value(&under)});
		exit.condition = Value(&*exit.PrependNewInst(exit.end(), ValueOpcode::LogicalOr, {exit.condition, Value(&over)}));
	}
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

}

bool GuardLoop(Program& program, Block& header, uint32_t limit, bool mark) {
	const auto first = BlockAt(program, &header);
	const auto last  = BlockAt(program, header.terminator.continue_block);
	auto*      merge = header.terminator.merge_block;
	Block*     latch = nullptr;
	for (auto* pred: header.ImmPredecessors()) {
		const auto at = BlockAt(program, pred);
		if (at < first || at > last) continue;
		if (latch != nullptr) return false;
		latch = pred;
	}
	if (latch == nullptr || merge == nullptr || merge == &header) return false;
	auto&      term        = latch->terminator;
	const bool conditional = term.kind == CFG::TerminatorKind::ConditionalBranch && !latch->condition.IsEmpty() &&
	                         (term.true_block == &header || term.false_block == &header);
	const bool plain       = term.kind == CFG::TerminatorKind::Branch && term.true_block == &header &&
	                         std::ranges::find(latch->ImmSuccessors(), merge) == latch->ImmSuccessors().end();
	if (!conditional && !plain) return false;
	Block* exit = nullptr;
	Block* stay = nullptr;
	if (plain) {
		exit = EveryIterationExit(program, header, latch, first, last, stay);
		if (exit == nullptr) {
			if (BodyValuesLeave(program, first, last, merge)) return false;
			for (const auto& inst: *merge) {
				if (inst.GetOpcode() != ValueOpcode::Phi) break;
				if (!ZeroAvailable(inst.GetType())) return false;
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
	if (conditional) return LeaveAfter(*latch, header, Value(&next), limit, mark);
	if (exit != nullptr) return LeaveAfter(*exit, *stay, Value(&next), limit, mark);
	for (auto& inst: *merge) {
		if (inst.GetOpcode() != ValueOpcode::Phi) break;
		inst.AddPhiOperand(latch, ZeroOf(*latch, inst.GetType()));
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
	return true;
}

LoopGuards GuardComputeLoops(Program& program, uint32_t limit) {
	LoopGuards guards;
	if (program.dispatcher_fallback) {
		guards.unguarded = 1;
		return guards;
	}
	std::vector<Block*> headers;
	for (auto* block: program.blocks) {
		const auto& term = block->terminator;
		if (term.loop_header && term.continue_block != nullptr && term.merge_block != nullptr) headers.push_back(block);
	}
	for (auto* header: headers) {
		++(GuardLoop(program, *header, limit, true) ? guards.guarded : guards.unguarded);
	}
	return guards;
}

}

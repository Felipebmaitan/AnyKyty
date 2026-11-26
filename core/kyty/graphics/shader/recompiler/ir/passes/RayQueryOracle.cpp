#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr size_t MaxVisited = 4096;

bool CrossesLanes(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::WriteLane:
		case ValueOpcode::Ballot:
		case ValueOpcode::SubgroupUMinU32:
		case ValueOpcode::SubgroupUMaxU32:
		case ValueOpcode::SubgroupOrU32:
		case ValueOpcode::PermuteU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32: return true;
		default: return false;
	}
}

bool KeepsLane(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::Identity:
		case ValueOpcode::Phi:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::IAdd32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot: return true;
		default: return false;
	}
}

bool ChildrenStayInLane(std::span<Inst* const> walk) {
	std::unordered_set<const Inst*> seen(walk.begin(), walk.end());
	std::vector<const Inst*>        pending(walk.begin(), walk.end());
	while (!pending.empty()) {
		if (seen.size() > MaxVisited) return false;
		const auto* value = pending.back();
		pending.pop_back();
		for (const auto& use: value->Uses()) {
			const auto op = use.user->GetOpcode();
			if (CrossesLanes(op)) return false;
			if (op == ValueOpcode::MakeImageAddress || op == ValueOpcode::BvhIntersect) continue;
			if (!KeepsLane(op)) return false;
			if (seen.insert(use.user).second) pending.push_back(use.user);
		}
	}
	return true;
}

bool NodeFollowsChildren(const Inst& bvh, const std::unordered_set<const Inst*>& sites) {
	const auto* address = bvh.Arg(1).Resolve().TryInstruction();
	if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return false;
	std::unordered_set<const Inst*> seen;
	std::vector<Value>              pending {address->Arg(0)};
	bool                            fed = false;
	while (!pending.empty()) {
		if (seen.size() > MaxVisited) return false;
		const auto value = pending.back().Resolve();
		pending.pop_back();
		if (value.IsImmediate()) continue;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (!seen.insert(inst).second) continue;
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::CompositeExtractU32x4) {
			const auto* source = inst->Arg(0).Resolve().TryInstruction();
			if (source == nullptr || !sites.contains(source)) return false;
			fed = true;
			continue;
		}
		if (!KeepsLane(op)) return false;
		for (size_t arg = op == ValueOpcode::SelectU32 ? 1 : 0; arg < inst->NumArgs(); ++arg) {
			pending.push_back(inst->Arg(arg));
		}
	}
	return fed;
}

}

bool OracleLaneWalk(std::span<Inst* const> walk) {
	if (walk.empty() || !ChildrenStayInLane(walk)) return false;
	const std::unordered_set<const Inst*> sites(walk.begin(), walk.end());
	return std::ranges::all_of(walk, [&](const Inst* bvh) { return NodeFollowsChildren(*bvh, sites); });
}

uint32_t MarkOracleWalk(std::span<Inst* const> walk) {
	const uint32_t flags = BvhIntersectNative | BvhIntersectOracle | (OracleLaneWalk(walk) ? BvhIntersectOracleLane : 0u);
	for (auto* bvh: walk) {
		bvh->SetFlags(bvh->Flags<uint32_t>() | flags);
	}
	return walk.empty() ? 0u : flags;
}

namespace {

size_t BlockAt(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

Block* InnermostLoop(const Program& program, const Block* block) {
	const auto at = BlockAt(program, block);
	for (auto header = at + 1; header-- > 0;) {
		const auto& term = program.blocks[header]->terminator;
		if (term.loop_header && term.continue_block != nullptr && term.merge_block != nullptr &&
		    BlockAt(program, term.continue_block) >= at) {
			return program.blocks[header];
		}
	}
	return nullptr;
}

bool BoundLoop(Program& program, Block& header, uint32_t limit) {
	const auto first = BlockAt(program, &header);
	const auto last  = BlockAt(program, header.terminator.continue_block);
	Block*     latch = nullptr;
	for (auto* pred: header.ImmPredecessors()) {
		const auto at = BlockAt(program, pred);
		if (at < first || at > last) continue;
		if (latch != nullptr) return false;
		latch = pred;
	}
	if (latch == nullptr || latch->condition.IsEmpty() ||
	    latch->terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
		return false;
	}
	const bool repeat_on_true = latch->terminator.true_block == &header;
	if (!repeat_on_true && latch->terminator.false_block != &header) return false;
	auto& count = *header.PrependNewInst(header.begin(), ValueOpcode::Phi);
	count.SetFlags(Type::U32);
	const auto body = std::ranges::find_if(header, [](const Inst& inst) { return inst.GetOpcode() != ValueOpcode::Phi; });
	auto&      next = *header.PrependNewInst(body, ValueOpcode::IAdd32, {Value(&count), Value(1u)});
	for (auto* pred: header.ImmPredecessors()) {
		count.AddPhiOperand(pred, pred == latch ? Value(&next) : Value(0u));
	}
	auto& under = *latch->PrependNewInst(latch->end(), ValueOpcode::ULessThan32, {Value(&next), Value(limit)});
	if (repeat_on_true) {
		latch->condition = Value(&*latch->PrependNewInst(latch->end(), ValueOpcode::LogicalAnd, {latch->condition, Value(&under)}));
	} else {
		auto& over       = *latch->PrependNewInst(latch->end(), ValueOpcode::LogicalNot, {Value(&under)});
		latch->condition = Value(&*latch->PrependNewInst(latch->end(), ValueOpcode::LogicalOr, {latch->condition, Value(&over)}));
	}
	return true;
}

}

size_t MarkWalksEmpty(Program& program) {
	size_t marked = 0;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::BvhIntersect) continue;
			const auto flags = inst.Flags<uint32_t>();
			if ((flags & BvhIntersectEmpty) != 0) continue;
			inst.SetFlags((flags & ~BvhIntersectOracle & ~BvhIntersectOracleLane) | BvhIntersectEmpty);
			marked += (flags & BvhIntersectNative) == 0 ? 1u : 0u;
		}
	}
	return marked;
}

size_t BoundEmptyWalks(Program& program, uint32_t limit) {
	std::vector<Block*> headers;
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::BvhIntersect) continue;
			const auto flags = inst.Flags<uint32_t>();
			if ((flags & BvhIntersectEmpty) == 0 || (flags & BvhIntersectNative) != 0) continue;
			auto* header = InnermostLoop(program, block);
			if (header != nullptr && std::ranges::find(headers, header) == headers.end()) headers.push_back(header);
		}
	}
	size_t bounded = 0;
	for (auto* header: headers) {
		bounded += BoundLoop(program, *header, limit) ? 1u : 0u;
	}
	return bounded;
}

size_t MarkEmptyWalk(const Program& program, std::span<Inst* const> intersections, const NativeTraversalOptions& options) {
	if (!options.rewrite) return 0;
	size_t marked = 0;
	for (auto* bvh: intersections) {
		if ((bvh->Flags<uint32_t>() & (BvhIntersectNative | BvhIntersectEmpty)) != 0) continue;
		bvh->SetFlags(bvh->Flags<uint32_t>() | BvhIntersectEmpty);
		++marked;
	}
	if (marked != 0) {
		LOGF("%s universal ray tracing: hash=0x%016llx runs %zu of %zu intersections empty: no form and no host query serves "
		     "them, their box nodes have no child\n",
		     program.stage == ShaderType::Compute ? "CS" : "PS", static_cast<unsigned long long>(program.shader_hash), marked,
		     intersections.size());
	}
	return marked;
}

}

#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct LoopGuards {
	uint32_t guarded   = 0;
	uint32_t unguarded = 0;
};

bool       GuardLoop(Program& program, Block& header, uint32_t limit, bool mark);
LoopGuards GuardComputeLoops(Program& program, uint32_t limit);

}

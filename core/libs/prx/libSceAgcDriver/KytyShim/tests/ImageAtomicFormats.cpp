#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdio>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler;

bool CheckAtomicImageMaterialization(Prospero::BufferFormat format, bool wide, bool expected) {
    IR::ResourcePlan plan;
    plan.stage = ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete = true;
    const std::array<uint32_t, 8> words {0x1000u, (static_cast<uint32_t>(format) << 20u) | (3u << 30u),
                                         15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
    IR::DescriptorSource source;
    source.dword_count = words.size();
    for (size_t i = 0; i < words.size(); ++i) {
        source.dwords[i] = IR::Value(words[i]);
    }
    plan.descriptor_sources.push_back(std::move(source));
    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Storage;
    image.dimension = Decoder::ImageDimension::Dim2D;
    image.atomic = true;
    image.atomic64 = wide;
    image.read = true;
    image.written = true;
    image.source = 0u;
    plan.info.images.push_back(image);
    IR::CompileSrtWalker(plan);
    IR::ResourceSnapshot snapshot;
    IR::ResourceSpecialization specialization;
    const bool accepted = IR::MaterializeResources(plan, {}, snapshot, specialization);
    if (accepted == expected && (!accepted || (specialization.images.size() == 1u &&
            specialization.images.front().numeric_class == Prospero::TextureNumericClass::Uint))) {
        return true;
    }
    std::fprintf(stderr, "atomic image format %u, width %u: expected %d, materialized %d\n",
                 static_cast<unsigned>(format), wide ? 64u : 32u, expected, accepted);
    return false;
}

}

int main() {
    using F = Libs::Graphics::Prospero::BufferFormat;
    bool passed = true;
    for (const auto format : {F::k32UInt, F::k32Float, F::k32SInt}) {
        passed &= CheckAtomicImageMaterialization(format, false, true);
        passed &= CheckAtomicImageMaterialization(format, true, false);
    }
    for (const auto format : {F::k32_32UInt, F::k32_32Float, F::k32_32SInt, F::k16_16_16_16Float}) {
        passed &= CheckAtomicImageMaterialization(format, true, true);
        passed &= CheckAtomicImageMaterialization(format, false, false);
    }
    for (const auto format : {F::k8UInt, F::k8_8_8_8UNorm, F::k32_32_32_32UInt, F::kBc1UNorm,
                              F::kFmask32_S8_F4, F::kFmask64_S16_F8}) {
        passed &= CheckAtomicImageMaterialization(format, false, false);
        passed &= CheckAtomicImageMaterialization(format, true, false);
    }
    return passed ? 0 : 1;
}

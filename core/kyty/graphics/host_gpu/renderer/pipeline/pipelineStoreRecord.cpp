#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLinkedStore.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <fmt/format.h>
#include <istream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics {
namespace {

void NoteStoreRefused(ShaderType stage, uint64_t hash,
                      const ShaderRecompiler::IR::EncodeFailure& failure) {
	using Reason = ShaderRecompiler::IR::EncodeFailure::Reason;
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		std::scoped_lock lock(mutex);
		if (!noted.insert(hash).second) {
			return;
		}
	}
	std::string why;
	switch (failure.reason) {
		case Reason::ValueType:
			why = fmt::format("a value of type 0x{:08x} has no cache encoding", failure.type);
			break;
		case Reason::UnknownInstruction:
			why = "a value refers to an instruction outside the program's plan";
			break;
		case Reason::PhiBlock: why = "a phi operand names a predecessor block"; break;
		case Reason::None: return;
	}
	const std::string_view opcode =
	    failure.in_instruction ? ShaderRecompiler::IR::ValueOpcodeName(failure.opcode)
	                           : std::string_view("none, outside the instructions");
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: program 0x{:016x} ({}) is not written to the store: {} (opcode {}); the game "
	    "goes on with the program compiled in memory\n",
	    hash, ShaderRecompiler::StageName(stage), why, opcode));
}

}

bool EncodeModuleRecord(ShaderType stage, uint64_t hash, uint32_t user_data_count,
                        const std::vector<uint32_t>&              static_state,
                        const ShaderRecompiler::IR::ResourcePlan& plan, const ShaderModule& module,
                        const std::vector<uint32_t>& spirv, const std::vector<uint32_t>& plain_spirv,
                        std::vector<uint8_t>& record) {
	using ShaderRecompiler::IR::IoAll;
	using ShaderRecompiler::IR::Writer;
	using namespace ShaderStoreFormat;

	std::vector<uint8_t>                plan_bytes;
	ShaderRecompiler::IR::EncodeFailure failure;
	if (!ShaderRecompiler::IR::SaveResourcePlan(plan, plan_bytes, failure)) {
		NoteStoreRefused(stage, hash, failure);
		return false;
	}
	Writer writer;
	IoAll(writer, Record::Module, stage, hash, user_data_count, static_state, plan_bytes);
	IoStoreModuleInfo(writer, module);
	IoAll(writer, spirv, plain_spirv);
	if (writer.failure.Failed()) {
		NoteStoreRefused(stage, hash, writer.failure);
		return false;
	}
	record = std::move(writer.bytes);
	return true;
}

#if defined(ANYPS5_KYTY)
bool LinkedComputeStore::Read(std::istream& input, std::span<const uint8_t> header) {
	using namespace ShaderStoreFormat;
	using namespace ShaderRecompiler::IR;
	const auto read = [&](void* data, size_t size) {
		return bool(input.read(static_cast<char*>(data), static_cast<std::streamsize>(size)));
	};
	std::array<char, 8> magic {};
	std::vector<uint8_t> profile(header.size());
	uint32_t count = 0;
	if (header.empty() || !read(magic.data(), magic.size()) || std::string_view(magic.data(), magic.size()) != "KYTYLCS1" ||
	    !read(profile.data(), profile.size()) || !std::ranges::equal(profile, header) || !read(&count, sizeof(count))) return false;
	decltype(m_entries) entries;
	std::unordered_set<uint64_t> ids;
	std::vector<uint8_t> compressed, raw;
	size_t budget = 256u << 20;
	for (uint32_t i = 0; i < count; ++i) {
		Frame frame;
		if (!read(&frame, sizeof(frame)) || !frame.raw_size || frame.raw_size > (64u << 20) ||
		    frame.raw_size > budget || frame.compressed_size > ZSTD_compressBound(frame.raw_size)) return false;
		compressed.resize(frame.compressed_size);
		if (!read(compressed.data(), compressed.size()) || !DecodeShaderStoreFrame(frame, compressed, raw)) return false;
		Reader reader {.bytes = raw, .allocation_budget = std::min<size_t>(64u << 20, budget)};
		const auto initial_budget = reader.allocation_budget;
		Entry entry;
		auto& key = entry.key;
		auto& module = entry.module;
		std::vector<uint8_t> plan;
		std::vector<uint32_t> plain;
		IoAll(reader, entry.caller, module.function_code);
		IoStoreModule(reader, key, module, plan, entry.spirv, plain);
		if (!reader.ok || !reader.bytes.empty() || !plain.empty() || plan.empty() ||
		    entry.caller.empty() || module.function_code.empty() || key.stage != ShaderType::Compute ||
		    module.info.stage != key.stage || module.info.shader_hash != key.hash || module.info.table_mode ||
		    module.mesh_task || !module.vertex_formats.empty() || !module.rect_parameters.empty() ||
		    entry.spirv.size() < 5 || entry.spirv.front() != 0x07230203u ||
		    XXH3_64bits(entry.caller.data(), entry.caller.size() * sizeof(uint32_t)) != key.hash) return false;
		Writer inputs;
		IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
		      module.info.bindings.push_data_start_dword, module.specialization);
		inputs.Raw(module.function_code.data(), module.function_code.size() * sizeof(uint32_t));
		if (inputs.failure.Failed() || XXH3_64bits(inputs.bytes.data(), inputs.bytes.size()) != module.id ||
		    !ids.insert(module.id).second) return false;
		budget -= std::max<size_t>(frame.raw_size, initial_budget - reader.allocation_budget);
		module.spirv_words = static_cast<uint32_t>(entry.spirv.size());
		entries.emplace(key.hash, std::move(entry));
	}
	if (input.peek() != std::char_traits<char>::eof() || input.bad()) return false;
	m_entries.swap(entries);
	return true;
}

const LinkedComputeStore::Entry* LinkedComputeStore::Find(
    uint64_t hash, uint32_t user_data_count, std::span<const uint32_t> static_state,
    std::span<const uint32_t> caller, std::span<const uint32_t> functions,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t cursor) const {
	const auto [first, last] = m_entries.equal_range(hash);
	for (auto it = first; it != last; ++it) {
		const auto& entry = it->second;
		const auto& layout = entry.module.info.bindings;
		if (entry.key.user_data_count == user_data_count && std::ranges::equal(entry.key.static_state, static_state) &&
		    std::ranges::equal(entry.caller, caller) && std::ranges::equal(entry.module.function_code, functions) &&
		    entry.module.specialization == specialization &&
		    layout.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(cursor, layout.ShaderDataDwords())) return &entry;
	}
	return nullptr;
}

bool LoadStoredComputeModule(std::istream& input, std::span<const uint8_t> expected_header,
                             uint64_t id, ShaderModule& module, std::vector<uint32_t>& spirv,
                             std::string& error) {
	using ShaderRecompiler::IR::IoAll;
	using ShaderRecompiler::IR::Reader;
	using namespace ShaderStoreFormat;

	constexpr size_t MaxRecordBytes = 64u * 1024u * 1024u;
	const auto fail = [&](const char* message) {
		error = message;
		return false;
	};
	const auto read = [&](void* data, size_t size) {
		return bool(input.read(static_cast<char*>(data), static_cast<std::streamsize>(size)));
	};
	std::vector<uint8_t> header(expected_header.size());
	if (header.empty() || !read(header.data(), header.size()) || !std::ranges::equal(header, expected_header)) {
		return fail("shader store is unreadable or incompatible with this Kyty build and device");
	}
	std::vector<uint8_t> compressed, raw;
	ShaderModule selected;
	std::vector<uint32_t> selected_spirv;
	while (input.peek() != std::char_traits<char>::eof()) {
		Frame frame;
		if (!read(&frame, sizeof(frame)) || frame.raw_size > MaxRecordBytes ||
		    frame.compressed_size > ZSTD_compressBound(MaxRecordBytes)) {
			return fail("invalid shader store frame size");
		}
		compressed.resize(frame.compressed_size);
		if (!read(compressed.data(), compressed.size()) || !DecodeShaderStoreFrame(frame, compressed, raw)) {
			return fail("invalid or incomplete shader store frame");
		}
		Reader reader {.bytes = raw, .allocation_budget = MaxRecordBytes};
		Record type {};
		IoAll(reader, type);
		if (!reader.ok || type > Record::TessellationLayout) {
			return fail("invalid shader store record kind");
		}
		if (type != Record::Module) continue;
		ShaderStoreFormat::Key key;
		ShaderModule candidate;
		std::vector<uint8_t> plan;
		std::vector<uint32_t> words, plain;
		IoStoreModule(reader, key, candidate, plan, words, plain);
		if (!reader.ok || !reader.bytes.empty()) return fail("invalid shader store module record");
		if (candidate.id != id) continue;
		if (!selected_spirv.empty()) return fail("duplicate module ID in shader store");
		if (key.stage != ShaderType::Compute || candidate.info.stage != ShaderType::Compute) {
			return fail("requested module is not a compute shader");
		}
		if (words.empty()) return fail("compute module has no SPIR-V");
		selected = std::move(candidate);
		selected_spirv = std::move(words);
	}
	if (input.bad() || !input.eof()) return fail("could not finish reading the shader store");
	if (selected_spirv.empty()) return fail("compute module ID was not found in the shader store");
	selected.spirv_words = static_cast<uint32_t>(selected_spirv.size());
	module = std::move(selected);
	spirv = std::move(selected_spirv);
	return true;
}
#endif

}

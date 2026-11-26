#include "graphics/host_gpu/renderer/pipeline/manifestStartup.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;
using namespace std::chrono_literals;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(1);
	}
}

void ModuleScheduling() {
	const std::array<uint32_t, 4> ranks {0, 0, 1, 1};
	const std::array<size_t, 4> words {2048, 4000, 16, 8000};
	std::deque<uint32_t> jobs {0, 1, 2, 3};
	const auto next = [&](bool allow_large, bool prefer_large) {
		return NextManifestModule(jobs, ranks, words, allow_large, prefer_large);
	};
	Check(*next(true, true) == 1, "a large worker overlaps work within the first rank");
	Check(*next(true, false) == 0, "a small pool preserves normal priority");
	Check(*next(false, false) == 0, "a 2048-word input qualifies as small");
	jobs.pop_front();
	Check(*next(false, false) == 2, "a small worker can pass a large job");
	jobs.erase(next(false, false));
	Check(next(false, false) == jobs.end(), "only large workers can take the remaining jobs");
}

void VideoWait() {
	const auto start = ManifestStartupWait::Clock::time_point {};
	Check(ManifestStartupWait(start, 0s).ReleaseReason(start) == "timed out", "zero wait releases immediately");
	Check(ManifestStartupWait(start, 5s).ReleaseReason(start + 5s) == "timed out", "a short user limit wins over the stall timeout");
	ManifestStartupWait wait(start, 900s);
	Check(wait.ReleaseReason(start + 119s).empty(), "a running job has time to finish");
	Check(wait.ReleaseReason(start + 120s) == "stalled", "a stalled job releases video");
	for (auto time = start + 100s; time < start + 900s; time += 100s) {
		wait.last_progress = time;
		Check(wait.ReleaseReason(time + 99s).empty(), "ongoing progress extends the stall deadline");
	}
	Check(wait.ReleaseReason(start + 900s) == "timed out", "progress cannot extend the maximum wait");
}

void MemoryPause() {
	constexpr uint64_t GiB = uint64_t {1} << 30;
	const auto now = ManifestMemoryPause::Clock::time_point {};
	const auto first = std::this_thread::get_id();
	std::thread second_thread([] {});
	const auto second = second_thread.get_id();
	second_thread.join();
	for (const auto total: {16 * GiB, 32 * GiB}) {
		const auto low = std::min(12 * GiB, total / 2);
		const auto high = std::min(14 * GiB, total / 12 * 7);
		ManifestMemoryPause memory;
		memory.Sample(now, total, low);
		Check(memory.Admit(now, first), "work starts at the low threshold");
		memory.Sample(now, total, low - 1);
		Check(!memory.Admit(now + 29s, first), "work pauses below the low threshold");
		memory.Sample(now + 30s, total, high);
		Check(memory.Admit(now + 30s, first), "a persistent pause admits one worker");
		Check(!memory.Admit(now + 31s, second), "other workers wait under pressure");
		Check(memory.Admit(now + 31s, first), "a worker can finish its nested pipeline work");
		memory.Finished(first);
		Check(memory.Admit(now + 31s, second), "the next worker can make progress");
		memory.Sample(now + 32s, total, high + 1);
		Check(memory.Admit(now + 32s, first), "all workers resume above the high threshold");
	}
}

}

int main() {
	ModuleScheduling();
	VideoWait();
	MemoryPause();
	return 0;
}

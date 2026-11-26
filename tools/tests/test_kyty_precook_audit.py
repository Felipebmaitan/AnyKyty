from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kyty_precook_audit import AUDIT_ENABLED, summarize


VARIANT = "0123456789abcdef"
START = "[kyty-startup] guest_entry monotonic_ns=100"
DRAW = "[kyty-startup] first_draw monotonic_ns=110"


def probe(mode, event, time):
    result = {"no_compile": "Success", "compile_required": "PipelineCompileRequired", "unverified": "ErrorUnknown"}[event]
    return (f"[kyty-driver-pipeline] kind=compute variant={VARIANT} mode={mode} "
            f"event={event} result={result} monotonic_ns={time}")


class PrecookAuditTests(unittest.TestCase):
    def report(self, *lines, enabled=True):
        if enabled:
            lines = (AUDIT_ENABLED, *lines)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stderr.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return summarize(path)

    def test_requires_vulkan_evidence(self):
        self.assertFalse(self.report(START, DRAW)["zero_compilation_observed"])
        self.assertTrue(self.report(probe("create", "compile_required", 90), START, DRAW)["zero_compilation_observed"])
        self.assertFalse(self.report(probe("create", "no_compile", 90), START, DRAW, enabled=False)["zero_compilation_observed"])

    def test_lookup_miss_differs_from_a_create_requiring_compilation(self):
        allowed = self.report(START, DRAW, probe("lookup", "compile_required", 121), probe("create", "no_compile", 122))
        self.assertTrue(allowed["zero_compilation_observed"])
        self.assertEqual(allowed["guest_creates_requiring_compilation"], 0)
        omitted = self.report(START, DRAW, probe("lookup", "compile_required", 121), probe("create", "compile_required", 122))
        self.assertFalse(omitted["zero_compilation_observed"])
        self.assertEqual(omitted["guest_compile_required_variants"], [f"compute/{VARIANT}"])

    def test_window_uses_timestamps_and_checks_shader_builds_skips_and_unverified(self):
        ready = probe("create", "no_compile", 80)
        build = f"[kyty-variant] build stage=CS module={VARIANT} @125"
        skip = f"[kyty-startup] runtime_pipeline kind=graphics route=skipped variant={VARIANT} shader={VARIANT} monotonic_ns=126"
        for event in (build, skip, probe("create", "unverified", 127)):
            with self.subTest(event=event):
                self.assertFalse(self.report(event, ready, START, DRAW)["zero_compilation_observed"])

    def test_rejects_combined_runs_and_incomplete_audit_rows(self):
        for lines in ((START, START), (START, "[kyty-driver-pipeline] incomplete")):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                self.report(*lines)


if __name__ == "__main__":
    unittest.main()

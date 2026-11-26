import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graphics-precache"))
import bundle


def words(*values):
    return struct.pack(f"<{len(values)}I", *values)


def sized(data):
    return words(len(data)) + data


def reflect():
    return sized(struct.pack("<4H", 0, 0, 0, 8))


def shader(kind, code):
    header = bytearray(96)
    struct.pack_into("<2I", header, 0, 0x34333231, 24)
    struct.pack_into("<2I", header, 64, 96, 64)
    header[90] = kind
    footer = bytearray(48)
    footer[:8] = b"barefoot"
    struct.pack_into("<4I", footer, 16, 1, 4, 0, 0)
    program = b"sgaProg\0" + words(96, 64) + header + words(code, 0, 0, 0) + footer
    return b"\x02a\0\x32" + sized(program) + bytes(152) + reflect()


def fixture(pixel_slot=1):
    header = b"SGD5" + bytes(8) + words(0) + reflect()
    shaders = words(1) + shader(2, 0x100) + words(1) + shader(1, 0x200) + words(0, 0, 0, 0)
    arrays = words(1, 0, 1, 0, 0, 0, 0, 0)
    raster = bytearray(16)
    raster[1] = 2
    depth = bytearray(24)
    depth[0] = 1
    blend = bytearray(68)
    blend[11] = 15
    states = words(1) + raster + words(1) + depth + words(1) + blend + words(0)
    row = bytearray(24)
    row[:2] = bytes([1, pixel_slot])
    row[9:11] = bytes([9, 15])
    row[18:22] = bytes([0, 255, 255, 255])
    struct.pack_into("<H", row, 16, 24)
    technique_and_pass = words(1, 1 << 24, 1) + row + words(4, 0, 0, 0, 0)
    binding_and_pool = words(0x80000000) + reflect() + reflect() + words(1, 0x80000000, 0) + sized(b"t\0")
    effect = b"\x02e\0" + words(0) + arrays + bytes(80) + states + technique_and_pass + binding_and_pool
    return header + shaders + words(1) + effect + bytes(10)


def fixture_hash(program, length):
    if length != 16 or program[4:] != bytes(12):
        raise ValueError("unexpected shader program extent")
    return struct.unpack_from("<I", program)[0]


class ShaderBundleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def read(self, data):
        path = self.root / "fixture.bin"
        path.write_bytes(data)
        return list(bundle.read_bundles(path, fixture_hash))

    def test_reports_stable_references_and_selected_states(self):
        bundles = self.read(fixture())
        summary, pairs = bundle.summarize(bundles)
        self.assertEqual((summary["bundles"], summary["techniques"], summary["graphics_passes"]), (1, 1, 1))
        self.assertEqual(pairs, {("0000000000000100", "0000000000000200")})
        effect, = bundles[0]["effects"]
        row, = effect["passes"]
        self.assertEqual(row["shaders"]["vs"]["native_kind"], 2)
        self.assertEqual(row["states"], dict(rasterizer=0, depth_stencil=0, blend=0))
        self.assertEqual(row["bindings"], [0, None, None, None])
        self.assertEqual(row["stencil_reference"], 9)
        self.assertEqual(row["usage_offset"], 0)
        self.assertEqual(effect["rasterizer"][0]["cull_mode"], 2)
        self.assertEqual(effect["depth_stencil"][0]["depth_enable"], 1)
        self.assertEqual(effect["blend"][0]["targets"][0]["write_mask"], 15)
        manifest = self.root / "fixture.kpm"
        manifest.write_text("M a vs code=0000000000000100\nM b ps code=0000000000000200\n"
                            "M c ps code=0000000000000300\nP v=a p=b\nP v=a p=c\n")
        self.assertEqual(bundle.compare_manifest(manifest, pairs), dict(graphics_records=2, with_shipped_stage_pair=1))

    def test_reads_two_aligned_bundles(self):
        data = bytearray(3 * bundle.ALIGNMENT)
        for offset in (bundle.ALIGNMENT, 2 * bundle.ALIGNMENT):
            member = fixture()
            data[offset:offset + len(member)] = member
        self.assertEqual([group["offset"] for group in self.read(data)], [4096, 8192])

    def test_rejects_bad_shader_slot_and_truncated_table(self):
        with self.assertRaisesRegex(ValueError, "missing shader variant"):
            self.read(fixture(pixel_slot=2))
        with self.assertRaisesRegex(ValueError, "exceeds the input"):
            self.read(fixture()[:-12])


if __name__ == "__main__":
    unittest.main()

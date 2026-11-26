import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def footer():
    trailer = bytearray(48)
    trailer[:8] = b'barefoot'
    struct.pack_into('<4I', trailer, 16, 1, 16, 0, 0)
    return trailer


def create_compute_shader():
    registers = [(0x207, 8), (0x208, 4), (0x209, 1), (0x212, 0), (0x213, 0), (0x20C, 0), (0x20D, 0)]
    header = bytearray(96 + len(registers) * 8 + 20)
    struct.pack_into('<2I', header, 0, 0x34333231, 24)
    struct.pack_into('<2q', header, 32, 96 - 32, 96 + len(registers) * 8 - 40)
    struct.pack_into('<2I', header, 64, len(header), 16 + len(footer()))
    header[92] = len(registers)
    for index, register in enumerate(registers):
        struct.pack_into('<2I', header, 96 + index * 8, *register)
    return bytes(header + struct.pack('<4I', 0xBF810000, 0, 0, 0) + footer())


def create_archive(shader):
    extra = len(shader) - 15
    compressed = b'\xf0' + b'\xff' * (extra // 255) + bytes([extra % 255]) + shader
    return (struct.pack('<5I', 2, 0, 0, 0, 1) + struct.pack('<QIIB', 0, len(compressed), len(shader), 0)
            + struct.pack('<2I', 0, 0) + compressed)


def run(binary, source):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        assets = work / 'assets'
        assets.mkdir()
        output = work / 'inventory'
        shader = create_compute_shader()
        (assets / 'pack.ushaderbytecode').write_bytes(create_archive(shader))
        (assets / 'duplicate.bin').write_bytes(shader[-64:])
        (assets / 'untyped.bin').write_bytes(struct.pack('<4I', 0xBF810000, 1, 0, 0) + footer())
        (assets / 'unreadable.ushaderbytecode').write_bytes(struct.pack('<I', 1))
        wrapper = source / 'tools/precache/precache'
        subprocess.run([sys.executable, wrapper, '--binary', binary, 'inventory', assets, output], check=True)
        counts = json.loads((output / 'report.json').read_text())
        assert counts['shader_programs'] == 2, counts
        assert counts['creation_records'] == 1, counts
        assert counts['unresolved_containers'] == 1, counts
        assert (output / 'creation-records.bin').read_bytes() == shader
        store = output / 'shaders.code'
        assert store.read_bytes().startswith(struct.pack('<Q', 0x3145444F4359544B))
        recipes = output / 'compute-recipes.txt'
        marker, recipe = recipes.read_text().splitlines()
        digest, *fields = recipe.split()
        assert marker == 'kyty compute recipes 1' and len(digest) == 16
        assert fields == ['8', '4', '1', '0', '0', '0', '0'], recipe
        request = struct.pack('<Q8I', int(digest, 16), *map(int, fields), 4) + shader[-64:-48]
        subprocess.run([binary, '--prepare-compute'], input=request, cwd=work, check=True)
        assert 'cs code=' + digest in (work / 'compute.kpm').read_text()
        assert (work / 'compute-0.spv').read_bytes().startswith(struct.pack('<I', 0x07230203))
        failed = subprocess.run([binary, '--inventory', assets, output], capture_output=True)
        assert failed.returncode != 0
        assert (output / 'creation-records.bin').read_bytes() == shader


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parents[2])

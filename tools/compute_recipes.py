import argparse
import ctypes
import ctypes.util
import mmap
import os
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def code_index(path):
    records = {}
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if stream.read(8) != struct.pack("<Q", 0x3145444F4359544B):
            raise ValueError("unsupported code store")
        while header := stream.read(16):
            if len(header) != 16:
                raise ValueError("truncated code record")
            digest, words, compressed = struct.unpack("<QII", header)
            if not words or not compressed or compressed > size - stream.tell():
                raise ValueError("invalid code length")
            stream.seek(compressed, os.SEEK_CUR)
            if digest and records.setdefault(digest, words) != words:
                raise ValueError("inconsistent code length")
    return records


def read_recipe(data, header, header_size):
    end = header + header_size
    registers = header + 32 + struct.unpack_from("<q", data, header + 32)[0]
    specials = header + 40 + struct.unpack_from("<q", data, header + 40)[0]
    count = data[header + 92]
    if not header + 96 <= registers <= end - 8 * count or not header + 96 <= specials <= end - 20:
        return None
    values = dict(struct.iter_unpack("<II", data[registers:registers + 8 * count]))
    required = (0x207, 0x208, 0x209, 0x212, 0x213)
    if not all(key in values for key in required) or values.get(0x20C) != 0 or values.get(0x20D, 1) & 0xff:
        return None
    group = tuple(values[key] for key in required[:3])
    if not all(group) or group[0] * group[1] * group[2] > 1024:
        return None
    scratch = struct.unpack_from("<H", data, header + 84)[0]
    modifier = struct.unpack_from("<I", data, specials + 16)[0]
    return (*group, values[0x212], values[0x213], modifier, scratch)


def scan(path, index, hash_code, stages, recipes):
    magic = struct.pack("<II", 0x34333231, 24)
    with path.open("rb") as file:
        if not os.fstat(file.fileno()).st_size:
            return
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            cursor = 0
            while (header := data.find(magic, cursor)) >= 0:
                cursor = header + len(magic)
                if header + 96 > len(data):
                    continue
                header_size, size = struct.unpack_from("<II", data, header + 64)
                code = header + header_size
                footer = code + size - 48
                if header_size < 96 or size < 48 or code + size > len(data) or data[footer:footer + 8] != b"barefoot":
                    continue
                version, length, constants, metadata = struct.unpack_from("<4I", data, footer + 16)
                words_size = ((length + 15) & ~15) + constants
                if version != 1 or not length or words_size % 4:
                    continue
                if ((words_size + 15) & ~15) + ((metadata + 15) & ~15) != size - 48:
                    continue
                digest = hash_code(data[code:code + words_size], words_size)
                if index.get(digest) != words_size // 4:
                    continue
                stage = data[header + 90]
                stages[digest].add(stage)
                if stage == 0:
                    recipe = read_recipe(data, header, header_size)
                    if recipe is not None:
                        recipes.add((digest, *recipe))


def main():
    parser = argparse.ArgumentParser(description="Index contiguous shader creation headers by verified program hash")
    parser.add_argument("store", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in [args.store, *args.files]}:
        parser.error("the output must be distinct from its inputs")
    lib = ctypes.CDLL(ctypes.util.find_library("xxhash") or "libxxhash.so.0")
    hash_code = lib.XXH3_64bits
    hash_code.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    hash_code.restype = ctypes.c_uint64
    index = code_index(args.store)
    stages = defaultdict(set)
    recipes = set()
    for path in args.files:
        scan(path, index, hash_code, stages, recipes)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as out:
        temp = Path(out.name)
        try:
            print("kyty compute recipes 1", file=out)
            for digest, *values in sorted(recipes):
                print(f"{digest:016x}", *values, file=out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temp, args.output)
        finally:
            temp.unlink(missing_ok=True)
    counts = Counter(stage for found in stages.values() for stage in found)
    print(f"{len(stages)}/{len(index)} programs with creation headers, {counts[0]} compute, "
          f"{len({r[0] for r in recipes})} with complete inputs, {len(recipes)} recipes")


if __name__ == "__main__":
    main()

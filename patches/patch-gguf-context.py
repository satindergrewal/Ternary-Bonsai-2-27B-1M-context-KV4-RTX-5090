#!/usr/bin/env python3
"""Patch <arch>.context_length in a GGUF header (single u32, in place).

Why: llama.cpp llama-server caps slot context at the model's trained context
(n_ctx_train, derived from the GGUF context_length key) - a larger -c is
silently re-capped per slot. To serve a 262K-native model at 1M with
--yarn-orig-ctx, flip that one header value to 1048576.

Weights are untouched: this edits exactly 4 bytes of metadata. To revert,
re-run with the original value (262144 for the official Bonsai 2 27B GGUFs)
or re-download the file. Stop any server using the file first.

Usage:
  python3 patch-gguf-context.py MODEL.gguf --check
  python3 patch-gguf-context.py MODEL.gguf 1048576
"""
import struct, sys

SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def find_context_length(path):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    if f.read(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    ver, nt, nk = u32(), u64(), u64()
    found = None
    for _ in range(nk):
        n = u64(); key = f.read(n).decode(errors="replace")
        t = u32()
        if t == 4:
            voff = f.tell(); v = u32()
            if key.endswith(".context_length"):
                found = (key, voff, v)
        elif t == 8:
            m = u64(); f.read(m)
        elif t == 9:
            et = u32(); n2 = u64()
            if et == 8:
                for _ in range(n2):
                    m = u64(); f.read(m)
            else:
                f.seek(SIZES[et] * n2, 1)
        elif t in SIZES:
            f.read(SIZES[t])
        else:
            raise SystemExit("unsupported kv type %d" % t)
    f.close()
    if found is None:
        raise SystemExit("no *.context_length key found in header")
    return found

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    path = sys.argv[1]
    key, off, cur = find_context_length(path)
    print("%s: %s = %d (header offset %d)" % (path, key, cur, off))
    if sys.argv[2] == "--check":
        return
    new = int(sys.argv[2])
    if new == cur:
        print("already at target value, nothing to do")
        return
    if new >= 2**32:
        raise SystemExit("value does not fit u32")
    f = open(path, "r+b")
    f.seek(off); f.write(struct.pack("<I", new)); f.close()
    key2, off2, now = find_context_length(path)
    assert now == new
    print("patched: %s = %d (was %d; re-run with %d to revert)" % (key, now, cur, cur))

if __name__ == "__main__":
    main()

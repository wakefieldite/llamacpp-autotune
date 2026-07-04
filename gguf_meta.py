#!/usr/bin/env python3
"""
Minimal GGUF header metadata reader — stdlib only, no dependencies.

Reads only the KV metadata header (architecture, name, whether an embedded
chat template exists, and the MoE expert count) and stops before tensor
data, so it works instantly even on multi-GB files. Used by llm-autotune.sh
to derive real, file-sourced facts (MoE vs dense, --jinja, sampling quirks)
instead of asking the user to guess them.

Output: "architecture|display_name|has_chat_template(yes/no)|expert_count"
  e.g.  "deepseek2|GLM 4.7 Flash|yes|64"   (MoE, 64 experts)
        "llama|Meta-Llama-3.1-8B|yes|0"    (dense, expert_count 0)
On any parse failure, prints nothing and exits 1 — the caller treats that
as "skip flag recommendation / fall back to a full-file scan", never as a
fatal error.
"""
import struct
import sys

GGUF_TYPE_STR = 8
GGUF_TYPE_BOOL = 7
GGUF_TYPE_ARR = 9
SCALAR_FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
              6: '<f', 10: '<Q', 11: '<q', 12: '<d'}
SCALAR_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


def read_gguf_meta(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return None
        struct.unpack('<I', f.read(4))[0]          # version (unused)
        struct.unpack('<Q', f.read(8))[0]           # n_tensors (unused)
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def read_str():
            ln = struct.unpack('<Q', f.read(8))[0]
            return f.read(ln).decode('utf-8', errors='replace')

        def skip_or_read(t, want):
            if t == GGUF_TYPE_STR:
                return read_str()
            if t == GGUF_TYPE_BOOL:
                v = f.read(1)
                return (v != b'\x00') if want else None
            if t in SCALAR_FMT:
                v = f.read(SCALAR_SZ[t])
                return struct.unpack(SCALAR_FMT[t], v)[0] if want else None
            if t == GGUF_TYPE_ARR:
                at = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                vals = []
                for _ in range(n):
                    v = skip_or_read(at, want)
                    if want:
                        vals.append(v)
                return vals if want else None
            raise ValueError(f"unknown GGUF type id {t}")

        result = {}
        wanted = {'general.architecture', 'general.name',
                  'general.basename', 'tokenizer.chat_template'}
        expert_count = 0
        for _ in range(n_kv):
            key = read_str()
            vtype = struct.unpack('<I', f.read(4))[0]
            want = key in wanted or key.endswith('.expert_count')
            val = skip_or_read(vtype, want)
            if key.endswith('.expert_count') and isinstance(val, int):
                expert_count = val
            elif want:
                if key == 'tokenizer.chat_template':
                    result['has_chat_template'] = bool(val)
                else:
                    result[key] = val
        result['expert_count'] = expert_count
        return result


def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    try:
        m = read_gguf_meta(sys.argv[1])
        if m is None:
            sys.exit(1)
        arch = m.get('general.architecture', 'unknown')
        name = m.get('general.name', m.get('general.basename', ''))
        has_tmpl = 'yes' if m.get('has_chat_template') else 'no'
        expert_count = m.get('expert_count', 0)
        print(f"{arch}|{name}|{has_tmpl}|{expert_count}")
    except Exception:
        sys.exit(1)


if __name__ == '__main__':
    main()

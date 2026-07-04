# llamacpp-autotune
Empirically find the fastest stable llama.cpp server config for any GGUF model on your hardware — validated by real server loads, not just benchmarks. CUDA/ROCm/Vulkan/SYCL/CPU.

`llm-autotune.sh` benchmarks a single GGUF model against your actual hardware and emits a verified
`llama-server` command, measured throughput numbers, a
[llama-swap](https://github.com/mostlygeek/llama-swap) snippet, and a machine-readable
`SUMMARY.json`. Every configuration it recommends is validated by launching a real server at your
target context and forcing a real decode — not just inferred from a quick benchmark that might crash
the moment you load a full context.

It is hardware-agnostic: devices and VRAM are read from your llama.cpp build's own `--list-devices`,
so the same script works unmodified on NVIDIA (CUDA), AMD (ROCm/HIP), Intel Arc (SYCL), Vulkan, or
CPU-only builds.

---

## Why this exists

llama.cpp gives you a lot of knobs — `-ngl`, `-ncmoe`, `-fa`, `-ub`, `-t`, KV cache quantization,
`--jinja` — and the fastest *stable* combination is different for every model/GPU/context pairing.
The usual ways of finding it fall short in specific ways:

- **`llama-bench` passing at moderate depth does not prove a config loads at full context.** A
  configuration can benchmark fine at depth 4096 and then fail to allocate at your real 32k context,
  or crash on the first decode rather than at load. This tool validates every fit decision by
  launching the real `llama-server` at your target `-c`, waiting for `/health`, **and** forcing a
  real completion.
- **The MoE `-ncmoe` pass/fail boundary is stochastic, not a fixed line.** Allocator and
  fragmentation variance mean the same value can pass one run and fail the next. This tool treats
  margin and stability repetitions as explicit policy knobs, re-verifies with a full repetition
  count after every adjustment, and warns you in the summary when your final value sits right on the
  edge of an observed failure.
- **Existing tools solve half the problem.** llama.cpp's `--fit` finds a config that fits but
  doesn't optimize it; benchmark-based optimizers find a fast config but don't deploy-validate it.
  This tool does both: optimize against a stated objective, then prove the winner actually loads and
  runs.

Everything the script "knows" — MoE detection, `--jinja`, sampling quirks — is read from the GGUF
file itself or measured, never guessed.

---

## Requirements

- **Linux.** macOS and Windows are explicitly out of scope. (Arch Linux is first-class: missing
  dependencies get an interactive `pacman` prompt. Other distros get a list of what to install.)
- **A built llama.cpp** with `llama-server` and `llama-bench` in the same bin directory. For MoE
  models the build must support `-ncmoe` (`git pull && rebuild` if yours doesn't).
- **Python 3** and **curl** (plus `lscpu`, `nproc` — standard on any distro).
- Optional: `nvidia-smi` or AMD sysfs for peak-VRAM logging. If neither is present, tuning still
  works fully — VRAM logging is cosmetic; all fit decisions come from real server probes, not
  telemetry.

Both files must live in the same directory:

```
llm-autotune.sh
gguf_meta.py      # dependency-free GGUF header parser, called by the script
```

---

## Quick start

```bash
chmod +x llm-autotune.sh

./llm-autotune.sh \
  -m ~/models/your-model/Model-Q4_K_M.gguf \
  -c 32768 \
  -B ~/llama.cpp/build/bin
```

That runs a full tune targeting a 32k context and writes everything to a timestamped
`./autotune_<model>_<timestamp>/` directory. Read `SUMMARY.md` for the launch command and measured
numbers.

---

## Usage

```
Usage: ./llm-autotune.sh -m MODEL.gguf [options]
  -m  path to GGUF model (required; first shard if split)
  -c  target context size to validate against (default: 16384)
  -B  llama.cpp bin dir with llama-server + llama-bench (default: ./build/bin)
  -o  output directory (default: ./autotune_<model>_<timestamp>)
  -t  comma list of thread counts to sweep (default: auto from CPU topology)
  -T  server load timeout seconds; raise for huge models on slow disks (default: 180)
  -O  objective: tg (generation speed), pp (prompt speed), balanced (default)
  -M  ncmoe safety margin above the observed floor (default: 1;
      use 2 on systems where you've seen boundary flakiness)
  -R  stability restarts the final config must survive (default: 3)
  -q  quick mode: skip mmap sweep, KV-quant report, and depth curve
```

### Choosing an objective (`-O`)

Generation (`tg`) and prompt-processing (`pp`) speed trade against each other through the
micro-batch size `-ub`. Rather than silently pick one, the tool optimizes a **stated** objective and
always prints the full candidate table so you can see the trade-off:

- `-O tg` — maximize token *generation* speed. Best for interactive chat where you wait on the model
  typing.
- `-O pp` — maximize *prompt ingestion* speed. Best for long-context / RAG / batch workloads where
  you feed large prompts.
- `-O balanced` *(default)* — a geometric-mean blend weighted toward generation.

### Tuning the safety margin (`-M`) and stability (`-R`)

For MoE models, the VRAM fit boundary is stochastic. `-M` sets how many `-ncmoe` layers of headroom
to keep above the observed failure floor (default 1). If you've seen a config load ten times and
fail the eleventh, run with `-M 2`. `-R` sets how many consecutive full-context restarts the final
config must survive before it's declared stable (default 3).

---

## Model types

The script auto-detects the model class from GGUF metadata and picks the right strategy:

- **MoE + GPU** (e.g. GLM-4.x-Flash, Qwen3-MoE): searches the `-ncmoe` CPU-offload floor per
  micro-batch candidate, so large expert models run on modest VRAM by keeping most experts in system
  RAM. This is the path with the stochastic-boundary handling.
- **Dense + GPU**: tries full GPU offload first; if the model is too large for VRAM at your target
  context, it searches for the maximum partial `-ngl` offload **at your requested context** rather
  than silently shrinking context first.
- **CPU-only**: threads become the primary lever; `-ncmoe` is irrelevant.

---

## Output

Each run produces a timestamped directory containing:

| File | Contents |
|------|----------|
| `SUMMARY.md` | Human-readable: final command, measured tg/pp, the candidate trade-off table, derived model flags with reasoning, and a llama-swap snippet. |
| `SUMMARY.json` | Machine-readable version of every decision and measurement — for aggregating results across models/machines. |
| `run.log` | Full timestamped log of every benchmark and probe. |
| `raw_*.jsonl` | Raw `llama-bench` JSONL for each phase, so you can re-analyze. |
| `probe_server.log` | Output from the most recent server fit-probe. |
| `meta.txt` | Host, CPU, RAM, and the llama.cpp build's device list. |

### Example `SUMMARY.md` excerpt

```markdown
- Decisions: fa=on ub=2048 b=2048 ncmoe=43 (observed floor: 41) t=12 mmap=on kv=f16 arch=deepseek2
## Measured performance (final config)
- tg128 @ d=4096: **18.13 t/s**   |   pp512 @ d=4096: **255.31 t/s**

## Final command
```bash
llama-server -m Model-Q4_K_M.gguf -ngl 99 -ncmoe 43 -fa on -c 32768 \
  -t 12 -b 2048 -ub 2048 --jinja --temp 1.0 --top-p 0.95 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port 8081
```
```

---

## Methodology notes

The header comment in `llm-autotune.sh` documents the hard-won lessons this tool encodes. A few
highlights:

- **KV cache quant types must be symmetric** (`ctk == ctv`).
- **`-ub` is the biggest prompt-processing lever for MoE hybrids** but raises the VRAM floor, so the
  floor is re-searched per micro-batch candidate (seeded from the previous rung — a smaller `-ub`
  never needs *more* CPU-offloaded layers).
- **Threads only matter when real work runs on CPU** (MoE offload or CPU-only builds); at full GPU
  offload the thread count is empirically flat, so that sweep is skipped.
- **`--jinja` is added automatically when the GGUF embeds its own chat template.** llama.cpp's
  built-in templates are a fixed, older set that can silently mis-render newer or custom templates —
  using the model's own template is strictly safer whenever one exists.
- **Sampling quirks are keyed on model name, not just architecture.** Distinct families share
  llama.cpp graph implementations (GLM-4.x MoE and DeepSeek-V2/V3 both report
  `architecture=deepseek2`), so architecture alone is not a reliable key. The quirk table is
  intentionally small — documented recommendations only.

---

## Limitations

- **Single-GPU only.** If multiple devices are present the first is used; multi-GPU (`-ts`/`-sm`)
  tuning is left to you.
- **Linux only**, by design.
- The sampling-quirk table covers only architectures with a *known, documented* reason to deviate
  from llama.cpp defaults. If you find one worth adding, the `SAMPLING_FLAGS` case statement is the
  place — contributions welcome.
- The emitted command uses `--port 8081` as a placeholder; change it if that port is occupied (it
  has no bearing on the tuning results).

---

## Contributing

Issues and PRs welcome, especially: new documented sampling quirks, additional backend coverage
reports (ROCm/SYCL/Vulkan results are valuable), and edge cases in GGUF metadata parsing.

## License

This project is dedicated to the public domain under [The Unlicense](LICENSE).

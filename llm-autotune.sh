#!/usr/bin/env bash
# ============================================================================
# llm-autotune.sh (v2) — empirical llama.cpp tuning + deploy validation
# ============================================================================
# Finds the fastest STABLE llama-server configuration for a GGUF model on a
# single-GPU Linux machine (any ggml backend: CUDA, ROCm/HIP, Vulkan, SYCL,
# or CPU-only), then emits a verified launch command, measured performance
# numbers, a llama-swap snippet, and a machine-readable SUMMARY.json.
#
# Platform scope: Linux only. Arch Linux is first-class (missing deps get an
# interactive pacman prompt); other distros get a list of required tools.
# macOS and Windows are explicitly unsupported.
#
# Hardware detection philosophy: the llama.cpp BUILD is the source of truth.
# Devices, VRAM totals and free memory are read from `llama-server
# --list-devices` (ggml's own view), so the same script works unmodified on
# NVIDIA, AMD, Intel Arc, or iGPU boxes. Vendor tools (nvidia-smi, amdgpu
# sysfs) are used only for optional peak-VRAM sampling.
#
# Methodology (hard-won lessons this script encodes):
#   1. llama-bench passing at moderate depth does NOT prove a config loads at
#      full context. All fit decisions are validated by launching the real
#      llama-server at the target -c, waiting for /health, AND forcing a real
#      decode (configs can crash at first decode, not at load).
#   2. The MoE -ncmoe pass/fail boundary is STOCHASTIC, not a fixed line:
#      allocator/fragmentation variance means the same value can pass one run
#      and fail the next. Therefore margin (-M) and stability repetitions
#      (-R) are explicit policy knobs, the stability loop re-verifies with
#      the full rep count after every bump, and the summary warns when the
#      final value sits adjacent to an observed failure.
#   3. KV cache quant types must be SYMMETRIC (ctk == ctv).
#   4. -ub is the biggest pp lever for MoE hybrids but raises the VRAM floor,
#      so the ncmoe floor is re-searched per ub candidate (seeded from the
#      previous rung: smaller ub never needs MORE CPU-MoE layers).
#   5. Speed is multi-objective: tg and pp trade against each other through
#      ub. The optimizer maximizes a STATED objective (-O tg|pp|balanced) and
#      always prints the full candidate table so the trade-off is visible.
#   6. Threads matter only when real work runs on CPU (MoE offload or
#      CPU-only builds); at full GPU offload they are empirically flat.
#   7. Warmup ON, -r >= 3, --delay between tests.
#   8. Final config gets a confirmation bench so the summary reports
#      measured numbers, not just decisions.
#
# Complements (does not replace): llama.cpp --fit (fits, doesn't optimize)
# and llama-optimus (optimizes via bench, doesn't deploy-validate).
# ============================================================================
set -uo pipefail

# ----------------------------- defaults ------------------------------------
CTX=16384
BINDIR="./build/bin"
OUTDIR=""
QUICK=0
THREADS_OVERRIDE=""
PROBE_PORT=18999
LOAD_TIMEOUT=180
REPS=3
MODEL=""
OBJECTIVE="balanced"   # tg | pp | balanced
MARGIN=1               # ncmoe safety margin above observed floor
STAB_REPS=3            # consecutive full-ctx restarts the final config must survive

usage() {
  cat <<EOF
Usage: $0 -m MODEL.gguf [options]
  -m  path to GGUF model (required; first shard if split)
  -c  target context size to validate against (default: $CTX)
  -B  llama.cpp bin dir with llama-server + llama-bench (default: $BINDIR)
  -o  output directory (default: ./autotune_<model>_<timestamp>)
  -t  comma list of thread counts to sweep (default: auto from CPU topology)
  -T  server load timeout seconds; raise for huge models on slow disks (default: $LOAD_TIMEOUT)
  -O  objective: tg (generation speed), pp (prompt speed), balanced (default)
  -M  ncmoe safety margin above the observed floor (default: $MARGIN;
      use 2 on systems where you've seen boundary flakiness)
  -R  stability restarts the final config must survive (default: $STAB_REPS)
  -q  quick mode: skip mmap sweep, KV-quant report, and depth curve
EOF
  exit 1
}

while getopts "m:c:B:o:t:T:O:M:R:qh" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    c) CTX="$OPTARG" ;;
    B) BINDIR="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    t) THREADS_OVERRIDE="$OPTARG" ;;
    T) LOAD_TIMEOUT="$OPTARG" ;;
    O) OBJECTIVE="$OPTARG" ;;
    M) MARGIN="$OPTARG" ;;
    R) STAB_REPS="$OPTARG" ;;
    q) QUICK=1 ;;
    *) usage ;;
  esac
done
[ -z "$MODEL" ] && usage
case "$OBJECTIVE" in tg|pp|balanced) ;; *) echo "ERROR: -O must be tg, pp, or balanced"; exit 1 ;; esac

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------- OS validation (Linux only) -----------------------
if [ "$(uname -s)" != "Linux" ]; then
  echo "ERROR: this script supports Linux only (detected: $(uname -s))."
  echo "macOS and Windows are out of scope."
  exit 1
fi
DISTRO_ID="unknown"; DISTRO_LIKE=""
if [ -r /etc/os-release ]; then
  . /etc/os-release
  DISTRO_ID="${ID:-unknown}"; DISTRO_LIKE="${ID_LIKE:-}"
fi
IS_ARCH=0
case " $DISTRO_ID $DISTRO_LIKE " in *" arch "*) IS_ARCH=1 ;; esac
echo "OS: Linux ($DISTRO_ID${DISTRO_LIKE:+, like: $DISTRO_LIKE}) — $([ $IS_ARCH -eq 1 ] && echo 'Arch family: pacman install prompts enabled' || echo 'non-Arch: missing tools will be listed only')"

# ------------------------- dependency handling ------------------------------
declare -A PKG=( [python3]=python [curl]=curl [lscpu]=util-linux [nproc]=coreutils )
MISSING=()
for tool in python3 curl lscpu nproc; do
  command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Missing required tools: ${MISSING[*]}"
  if [ $IS_ARCH -eq 1 ]; then
    PKGS=(); for t in "${MISSING[@]}"; do PKGS+=("${PKG[$t]}"); done
    echo "Install on Arch with: sudo pacman -S --needed ${PKGS[*]}"
    read -r -p "Run that now? [y/N] " ans
    if [ "${ans,,}" = "y" ]; then
      sudo pacman -S --needed "${PKGS[@]}" || { echo "Install failed."; exit 1; }
    else
      echo "Cannot continue without required tools."; exit 1
    fi
  else
    echo "Please install the equivalents of: ${MISSING[*]} (packages vary by distro)."
    exit 1
  fi
fi

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; exit 1; }
SERVER="$BINDIR/llama-server"; BENCH="$BINDIR/llama-bench"
[ -x "$SERVER" ] || { echo "ERROR: $SERVER not found/executable (point -B at your llama.cpp build/bin)"; exit 1; }
[ -x "$BENCH" ]  || { echo "ERROR: $BENCH not found/executable"; exit 1; }

MODELNAME=$(basename "$MODEL" .gguf)
TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-"./autotune_${MODELNAME}_${TS}"}
mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"
SUMMARY="$OUTDIR/SUMMARY.md"
SRV_LOG="$OUTDIR/probe_server.log"
SRV_PID=""; SAMPLER_PID=""

log() {
  # IMPORTANT: must write to stderr, not stdout. Several functions
  # (decide_fa, ncmoe_floor, pick helpers) are called as `x=$(fn ...)` and
  # internally log via bench()/probe_server(). If log() wrote to stdout,
  # command substitution would capture every log line into the "return
  # value" (this was a real, observed bug: -fa ended up containing a
  # multi-line log blob instead of "on").
  local msg
  msg="[$(date +%H:%M:%S)] $*"
  echo "$msg" >&2
  echo "$msg" >> "$LOG"
}
cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null && wait "$SRV_PID" 2>/dev/null
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
}
trap cleanup EXIT

# free-port check for the probe server
while ! python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',$PROBE_PORT));s.close()" 2>/dev/null; do
  PROBE_PORT=$((PROBE_PORT+1))
done

# ------------------ build-capability probes (dialect tolerance) -------------
SRV_HELP=$("$SERVER" --help 2>&1 || true)
BEN_HELP=$("$BENCH" --help 2>&1 || true)
FIT_OFF=""
echo "$SRV_HELP" | grep -q -- '--fit ' && FIT_OFF="--fit off"
HAS_NCMOE=0
echo "$BEN_HELP" | grep -q -- '-ncmoe' && HAS_NCMOE=1
FA_ON="on"; FA_OFF="off"
echo "$BEN_HELP" | grep -q -- '--flash-attn <on' || { FA_ON="1"; FA_OFF="0"; }

# ------------------ device detection via the build itself -------------------
DEVLIST=$("$SERVER" --list-devices 2>&1 || true)
read -r GPUCOUNT GPU_NAME VRAM_TOTAL VRAM_FREE <<< "$(python3 - <<PY
import re
txt = """$DEVLIST"""
devs = re.findall(r'^\s*(\S+?):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)', txt, re.M)
devs = [d for d in devs if not d[0].upper().startswith("CPU")]
if devs:
    tag, name, tot, free = devs[0]
    print(len(devs), name.replace(" ", "_"), tot, free)
else:
    print("0 none 0 0")
PY
)"
GPU_NAME=${GPU_NAME//_/ }

# vendor telemetry for optional peak-VRAM sampling (best-effort only)
VRAM_SAMPLER="none"; AMD_SYSFS=""
if [ "$GPUCOUNT" -ge 1 ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    VRAM_SAMPLER="nvidia"
  else
    for f in /sys/class/drm/card*/device/mem_info_vram_used; do
      [ -r "$f" ] && { VRAM_SAMPLER="amdgpu_sysfs"; AMD_SYSFS="$f"; break; }
    done
  fi
  [ "$VRAM_SAMPLER" = "none" ] && log "Note: no VRAM telemetry available (fine — peak-VRAM logging disabled; all fit decisions use real server probes, not telemetry)."
fi

vram_used() {
  case "$VRAM_SAMPLER" in
    nvidia)        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 ;;
    amdgpu_sysfs)  python3 -c "print(int(open('$AMD_SYSFS').read())//1048576)" 2>/dev/null ;;
    *)             echo "" ;;
  esac
}

# ------------------------- hardware / model meta ----------------------------
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$PHYS" -ge 1 ] 2>/dev/null || PHYS=$(nproc)
LOGICAL=$(nproc)
{
  echo "date: $(date)"; echo "model: $MODEL"; echo "target ctx: $CTX"
  echo "objective: $OBJECTIVE  margin: $MARGIN  stability reps: $STAB_REPS"
  echo "distro: $DISTRO_ID (${DISTRO_LIKE:-n/a})"
  lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Core' | head -5
  free -h | head -2
  echo "devices (from llama.cpp build):"; echo "$DEVLIST"
} > "$OUTDIR/meta.txt" 2>&1

# ------------------- GGUF metadata: MoE detection + model flags -------------
# Full GGUF KV parser (reads the whole metadata section, not a truncated
# byte-range guess). Authoritative source for MoE detection (via
# <arch>.expert_count > 0) and the flag derivation below.
GGUF_INFO=$(python3 "$SCRIPTDIR/gguf_meta.py" "$MODEL" 2>/dev/null || echo "PARSE_FAIL")

ARCH="unknown"; MODEL_DISPLAY_NAME=""; HAS_CHAT_TEMPLATE="no"; EXPERT_COUNT=0
if [ "$GGUF_INFO" != "PARSE_FAIL" ] && [ -n "$GGUF_INFO" ]; then
  IFS='|' read -r ARCH MODEL_DISPLAY_NAME HAS_CHAT_TEMPLATE EXPERT_COUNT <<< "$GGUF_INFO"
fi
IS_MOE=0
[ "${EXPERT_COUNT:-0}" -gt 0 ] 2>/dev/null && IS_MOE=1
if [ "$GGUF_INFO" = "PARSE_FAIL" ]; then
  log "WARNING: GGUF metadata parse failed — falling back to a full-file expert_count scan (slower, but correct on any file size)."
  grep -aq "expert_count" "$MODEL" && IS_MOE=1
fi
log "GGUF metadata: arch=$ARCH name='${MODEL_DISPLAY_NAME:-n/a}' expert_count=${EXPERT_COUNT:-0} embedded_chat_template=$HAS_CHAT_TEMPLATE"

# --jinja rule: if the file embeds a chat template, use it. llama.cpp's
# built-in C++ templates are a fixed, older set that can silently mis-render
# newer/custom templates (observed with GLM-4.7-Flash: wrong output, no
# error, until --jinja was added). Using the model's own template is strictly
# safer whenever one exists.
JINJA_FLAG=""
[ "$HAS_CHAT_TEMPLATE" = "yes" ] && JINJA_FLAG="--jinja"

# Known sampling quirks. Architecture ALONE is not a reliable key: distinct
# families share llama.cpp graph implementations (DeepSeek-V2/V3 and
# GLM-4.x MoE both report general.architecture=deepseek2), so the model
# name is checked too. Intentionally small — documented quirks only.
NAME_LC=$(echo "$MODEL_DISPLAY_NAME" | tr '[:upper:]' '[:lower:]')
SAMPLING_FLAGS=""; SAMPLING_NOTE="llama.cpp defaults (no known quirk for this architecture/model)"
if [[ "$NAME_LC" == *glm* ]]; then
  SAMPLING_FLAGS="--temp 1.0 --top-p 0.95 --repeat-penalty 1.0"
  SAMPLING_NOTE="Z.ai/GLM-4.x recommended: repeat-penalty must be disabled (1.0) or GLM loops/degrades. (Reports as architecture=deepseek2; matched on model name.)"
elif [ "$ARCH" = "deepseek2" ]; then
  SAMPLING_NOTE="architecture=deepseek2 but name doesn't match known GLM pattern — likely real DeepSeek-V2/V3; check that model's card, defaults left as-is"
else
  case "$ARCH" in
    qwen3|qwen3moe)
      SAMPLING_FLAGS="--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0"
      SAMPLING_NOTE="Qwen3 model-card recommended sampling"
      ;;
    gptoss)
      SAMPLING_FLAGS="--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0"
      SAMPLING_NOTE="gpt-oss recommended: near-flat sampling, rely on the model's own calibration"
      ;;
  esac
fi
[ -n "$SAMPLING_FLAGS" ] && log "Sampling quirk detected: $SAMPLING_NOTE"
[ -n "$JINJA_FLAG" ] && log "Embedded chat template found -> adding --jinja"

if [ "$GPUCOUNT" -eq 0 ]; then MODE="cpu"
elif [ $IS_MOE -eq 1 ]; then MODE="moe"
else MODE="dense"; fi

log "Model: $MODELNAME | MoE: $([ $IS_MOE -eq 1 ] && echo yes || echo no) | mode: $MODE | objective: $OBJECTIVE"
log "CPU: $PHYS phys / $LOGICAL logical cores"
if [ "$GPUCOUNT" -ge 1 ]; then
  log "GPU: $GPU_NAME — ${VRAM_TOTAL} MiB total, ${VRAM_FREE} MiB free (per llama.cpp)"
  [ "$GPUCOUNT" -gt 1 ] && log "WARNING: $GPUCOUNT devices found. This script tunes for SINGLE GPU; the first device will be used. For multi-GPU, tune -ts/-sm manually."
  RESIDUAL=$(( VRAM_TOTAL - VRAM_FREE ))
  if [ "$RESIDUAL" -gt 300 ]; then
    log "WARNING: ${RESIDUAL} MiB of VRAM is already in use by other processes. Fit floors found now will be TIGHTER than a clean system — close GUI apps / browsers, or accept that the result bakes in this overhead."
  fi
else
  log "No GPU devices in this llama.cpp build — CPU-only tuning path."
  [ $IS_MOE -eq 1 ] && log "(MoE on CPU: ncmoe is irrelevant; threads and ub are the levers.)"
fi
if [ $IS_MOE -eq 1 ] && [ "$MODE" = "moe" ] && [ $HAS_NCMOE -eq 0 ]; then
  log "FATAL: MoE model but this llama.cpp build lacks -ncmoe. Update llama.cpp (git pull && rebuild)."
  exit 2
fi

D_MED=$(( CTX / 4 )); [ $D_MED -gt 4096 ] && D_MED=4096; [ $D_MED -lt 1024 ] && D_MED=1024

UB_LADDER="2048 1024 512"
[ "${VRAM_FREE:-0}" -ge 14000 ] && UB_LADDER="4096 2048 1024 512"

# ------------------------------ helpers -------------------------------------
get_ts() { # file, kind(pp|tg) -> first matching avg_ts
  python3 - "$1" "$2" <<'PY'
import json,sys
f,kind=sys.argv[1],sys.argv[2]
try:
    for line in open(f):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if kind=="pp" and r.get("n_gen",0)==0 and r.get("n_prompt",0)>0: print(f"{r['avg_ts']:.2f}"); break
        if kind=="tg" and r.get("n_gen",0)>0: print(f"{r['avg_ts']:.2f}"); break
except Exception: pass
PY
}

jsonl_summary() { # file -> one line describing every result row
  python3 - "$1" <<'PY'
import json,sys
rows=[]
try:
    for line in open(sys.argv[1]):
        line=line.strip()
        if line: rows.append(json.loads(line))
except Exception: pass
if not rows:
    print("no results"); raise SystemExit
def varies(k): return len({str(r.get(k)) for r in rows})>1
extra=[k for k in ("n_threads","n_ubatch","flash_attn","use_mmap","type_k","n_cpu_moe") if varies(k)]
short={"n_threads":"t","n_ubatch":"ub","flash_attn":"fa","use_mmap":"mmap","type_k":"kv","n_cpu_moe":"ncmoe"}
out=[]
for r in rows:
    t=f"pp{r['n_prompt']}" if r.get("n_gen",0)==0 else f"tg{r['n_gen']}"
    if r.get("n_depth",0): t+=f"@d{r['n_depth']}"
    tag="".join(f"({short[k]}={r.get(k)})" for k in extra)
    out.append(f"{t}{tag}={r['avg_ts']:.2f}")
print("  ".join(out))
PY
}

bench() { # tag, args...
  local tag="$1"; shift
  local out="$OUTDIR/raw_${tag}.jsonl"
  log "bench[$tag]: $*"
  "$BENCH" -m "$MODEL" -r "$REPS" --delay 2 -o jsonl "$@" \
      > "$out" 2>"$OUTDIR/raw_${tag}.err"
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then log "bench[$tag]: FAILED"; return 1; fi
  log "bench[$tag]: $(jsonl_summary "$out") t/s"
  return 0
}

probe_server() { # extra flags...
  local peakf="$OUTDIR/.vram_peak"
  : > "$SRV_LOG"; : > "$peakf"
  # shellcheck disable=SC2086
  "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PROBE_PORT" -c "$CTX" \
    $FIT_OFF "$@" >> "$SRV_LOG" 2>&1 &
  SRV_PID=$!
  if [ "$VRAM_SAMPLER" != "none" ]; then
    ( m=0; while kill -0 $SRV_PID 2>/dev/null; do
        v=$(vram_used); [ -n "$v" ] && [ "$v" -gt "$m" ] 2>/dev/null && m=$v
        echo "$m" > "$peakf"; sleep 1
      done ) & SAMPLER_PID=$!
  fi
  local ok=1 i=0
  while [ $i -lt "$LOAD_TIMEOUT" ]; do
    if ! kill -0 $SRV_PID 2>/dev/null; then ok=1; break; fi
    if curl -sf "http://127.0.0.1:$PROBE_PORT/health" >/dev/null 2>&1; then
      if curl -sf -m 120 "http://127.0.0.1:$PROBE_PORT/completion" \
           -H 'Content-Type: application/json' \
           -d '{"prompt":"Hello","n_predict":8}' >/dev/null 2>&1; then ok=0; fi
      break
    fi
    sleep 1; i=$((i+1))
  done
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
  local peak=""; [ -s "$peakf" ] && peak=" (peak VRAM $(cat "$peakf") MiB)"
  if [ $ok -eq 0 ]; then
    log "probe [$*]: PASS$peak"
  else
    log "probe [$*]: FAIL$peak — $(grep -iE 'error|out of memory|failed' "$SRV_LOG" | tail -1)"
  fi
  return $ok
}

ncmoe_floor() { # hi_seed, extra flags... -> echoes minimal passing ncmoe, or "none"
  # hi_seed lets successive ub rungs start from the previous floor: with a
  # DESCENDING ub ladder, a smaller ub never needs MORE CPU-MoE layers, so
  # the previous floor is a valid upper bound (verified, not assumed — if it
  # unexpectedly fails, fall back to 99).
  local hi="$1"; shift
  if ! probe_server -ngl 99 -ncmoe "$hi" "$@"; then
    if [ "$hi" -ge 99 ]; then echo "none"; return; fi
    hi=99
    probe_server -ngl 99 -ncmoe 99 "$@" || { echo "none"; return; }
  fi
  probe_server -ngl 99 -ncmoe 0 "$@" && { echo 0; return; }
  local lo=0 mid
  while [ $((hi - lo)) -gt 1 ]; do
    mid=$(( (lo + hi) / 2 ))
    if probe_server -ngl 99 -ncmoe "$mid" "$@"; then hi=$mid; else lo=$mid; fi
  done
  echo "$hi"
}

ngl_ceiling() { # extra flags... -> echoes MAXIMUM passing -ngl (0-99), or "none"
  # Partial GPU offload for DENSE models too large to fully fit in VRAM.
  # Opposite monotonic direction from ncmoe_floor: here MORE ngl = MORE VRAM
  # used (more layers pushed onto the GPU), so "fits" is true for low ngl and
  # flips to false above some ceiling — we binary-search for that ceiling
  # rather than a floor. Used only when -ngl 99 (full offload) does not fit
  # at the target context; keeps the user's requested -c intact instead of
  # silently shrinking context first, mirroring the MoE path's philosophy
  # that context is precious and VRAM allocation is the thing to negotiate.
  if ! probe_server -ngl 0 "$@"; then echo "none"; return; fi
  if probe_server -ngl 99 "$@"; then echo 99; return; fi
  local lo=0 hi=99 mid
  while [ $((hi - lo)) -gt 1 ]; do
    mid=$(( (lo + hi) / 2 ))
    if probe_server -ngl "$mid" "$@"; then lo=$mid; else hi=$mid; fi
  done
  echo "$lo"
}

pick_threads_list() {
  if [ -n "$THREADS_OVERRIDE" ]; then echo "$THREADS_OVERRIDE"; return; fi
  python3 - "$PHYS" "$LOGICAL" <<'PY'
import sys
p,l=int(sys.argv[1]),int(sys.argv[2])
c=sorted(set(x for x in [max(2,p-2),p,p+2,(p+l)//2,l-2,l] if 2<=x<=l))
print(",".join(map(str,c[:6])))
PY
}

pick_best_threads() { # jsonl_file
  python3 - "$1" "$PHYS" <<'PY'
import json,sys
best=(0,int(sys.argv[2])); vals=[]
try:
    for line in open(sys.argv[1]):
        r=json.loads(line)
        if r.get("n_gen",0)>0:
            vals.append((r["avg_ts"],r["n_threads"]))
            if r["avg_ts"]>best[0]: best=(r["avg_ts"],r["n_threads"])
except Exception: pass
if vals and (max(v for v,_ in vals)-min(v for v,_ in vals))/max(v for v,_ in vals) < 0.02:
    print(sys.argv[2])   # spread < 2% = noise: prefer physical-core count
else:
    print(best[1])
PY
}

pick_mmap() { # jsonl_file with -mmp 0,1 rows -> echoes 0 (no-mmap wins) or 1
  python3 - "$1" <<'PY'
import json,sys
s={0:[0,0],1:[0,0]}
try:
    for line in open(sys.argv[1]):
        r=json.loads(line); m=1 if r.get("use_mmap") in (True,1,"1") else 0
        if r.get("n_gen",0)>0: s[m][0]=r["avg_ts"]
        else: s[m][1]=r["avg_ts"]
except Exception: pass
print(0 if (s[0][0]+0.001*s[0][1]) >= (s[1][0]+0.001*s[1][1]) else 1)
PY
}

score_cfg() { # tg pp -> objective score (higher is better)
  python3 - "$OBJECTIVE" "$1" "$2" <<'PY'
import sys, math
obj = sys.argv[1]
tg = max(float(sys.argv[2] or 0), 1e-9)
pp = max(float(sys.argv[3] or 0), 1e-9)
if obj == "tg":   s = tg
elif obj == "pp": s = pp
else:             s = math.exp(0.7*math.log(tg) + 0.3*math.log(pp))  # balanced
print(f"{s:.4f}")
PY
}

gt() { python3 -c "import sys; print(1 if float('$1') > float('$2') else 0)"; }

decide_fa() { # two separate benches so a broken fa=on can't nuke the comparison
  local extra=("$@")
  local on_tg="" off_tg=""
  bench fa_on  "${extra[@]}" -fa "$FA_ON"  -p 512 -n 96 -d "$D_MED" && on_tg=$(get_ts "$OUTDIR/raw_fa_on.jsonl" tg)
  bench fa_off "${extra[@]}" -fa "$FA_OFF" -p 512 -n 96 -d "$D_MED" && off_tg=$(get_ts "$OUTDIR/raw_fa_off.jsonl" tg)
  # Guarded numeric compare: get_ts can return "" if a bench failed. Coerce
  # both sides through Python's float() inside a try, defaulting a missing or
  # non-numeric value to 0.0, so a failed fa=on bench never crashes the run
  # (it just loses the comparison, which is the safe outcome).
  python3 - "$FA_ON" "$FA_OFF" "$on_tg" "$off_tg" <<'PY'
import sys
fa_on, fa_off, on_s, off_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def num(s):
    try: return float(s)
    except (ValueError, TypeError): return 0.0
print(fa_on if num(on_s) >= num(off_s) else fa_off)
PY
}

# ============================================================================
FA="$FA_ON"; UB=512; B=2048; NC=""; T=$PHYS; NGL=99
MMAP_FLAG="-mmp 0"; SRV_MMAP="--no-mmap"; KV_ARGS=""; SRV_KV=""
BASE_FLAGS=(); FINAL_FLAGS=()
CAND_TABLE=""; OBSERVED_FLOOR=""

# assemble_final: single source of truth for the final flag set. Base flags
# (mode-specific) + derived flags (--jinja, sampling) are ALWAYS combined
# here — the v1 stability-retry path rebuilt flags by hand and silently
# dropped the derived ones; this prevents that class of bug.
assemble_final() {
  case "$MODE" in
    moe)
      BASE_FLAGS=(-ngl 99 -ncmoe "$NC" -fa "$FA" -c "$CTX" -t "$T" -b "$B" -ub "$UB")
      [ -n "$SRV_MMAP" ] && BASE_FLAGS+=("$SRV_MMAP")
      ;;
    dense)
      BASE_FLAGS=(-ngl "$NGL" -fa "$FA" -c "$CTX" -b 2048 -ub "$UB")
      # shellcheck disable=SC2206
      [ -n "$SRV_KV" ] && BASE_FLAGS+=($SRV_KV)
      ;;
    cpu)
      BASE_FLAGS=(-c "$CTX" -t "$T" -b 2048 -ub "$UB")
      [ -n "$SRV_MMAP" ] && BASE_FLAGS+=("$SRV_MMAP")
      ;;
  esac
  FINAL_FLAGS=("${BASE_FLAGS[@]}")
  [ -n "$JINJA_FLAG" ] && FINAL_FLAGS+=("$JINJA_FLAG")
  if [ -n "$SAMPLING_FLAGS" ]; then
    # shellcheck disable=SC2206
    local sarr=($SAMPLING_FLAGS)
    FINAL_FLAGS+=("${sarr[@]}")
  fi
}

case "$MODE" in
# ------------------------------ MoE + GPU -----------------------------------
moe)
  log "=== MoE path: fa -> (ub x ncmoe-floor via server probes) -> threads -> mmap ==="
  FA=$(decide_fa -ngl 99 -ncmoe 99 -t "$PHYS")
  log "flash-attn decision: $FA"

  BEST_SCORE=0; BEST_TG=0; BEST_PP=0; BEST_UB=0; BEST_NC=""; B_FINAL=2048
  PREV_FLOOR=99
  for ub in $UB_LADDER; do
    b=$(( ub > 2048 ? ub : 2048 ))
    log "--- ub=$ub: searching ncmoe floor at ctx=$CTX (seed hi=$PREV_FLOOR) ---"
    floor=$(ncmoe_floor "$PREV_FLOOR" -fa "$FA" -b "$b" -ub "$ub" -t "$PHYS" $SRV_MMAP)
    if [ "$floor" = "none" ]; then log "ub=$ub: does not fit at any ncmoe. Skipping."; continue; fi
    PREV_FLOOR=$floor
    nc=$(( floor + MARGIN )); [ $nc -gt 99 ] && nc=99
    log "ub=$ub: floor=$floor -> candidate ncmoe=$nc (margin +$MARGIN)"
    if bench "ub${ub}_nc${nc}" -ngl 99 -ncmoe "$nc" -fa "$FA" -b "$b" -ub "$ub" \
             $MMAP_FLAG -t "$PHYS" -p 2048 -n 96 -d "$D_MED"; then
      tg=$(get_ts "$OUTDIR/raw_ub${ub}_nc${nc}.jsonl" tg)
      pp=$(get_ts "$OUTDIR/raw_ub${ub}_nc${nc}.jsonl" pp)
      sc=$(score_cfg "${tg:-0}" "${pp:-0}")
      CAND_TABLE+="| $ub | $floor | $nc | ${tg:-?} | ${pp:-?} | $sc |"$'\n'
      log "candidate ub=$ub: tg=${tg:-?} pp=${pp:-?} score($OBJECTIVE)=$sc"
      if [ "$(gt "$sc" "$BEST_SCORE")" = "1" ]; then
        BEST_SCORE=$sc; BEST_TG=${tg:-0}; BEST_PP=${pp:-0}
        BEST_UB=$ub; BEST_NC=$nc; B_FINAL=$b; OBSERVED_FLOOR=$floor
      fi
    fi
  done
  [ -z "$BEST_NC" ] && { log "FATAL: nothing fits at ctx=$CTX. Reduce -c or use a smaller quant."; exit 2; }
  UB=$BEST_UB; NC=$BEST_NC; B=$B_FINAL
  log ">>> candidate trade-off table (objective=$OBJECTIVE):"
  log ">>> ub | floor | ncmoe | tg | pp | score"
  while IFS= read -r line; do [ -n "$line" ] && log ">>> $line"; done <<< "$CAND_TABLE"
  log ">>> chosen: ub=$UB b=$B ncmoe=$NC (tg=$BEST_TG pp=$BEST_PP). Different priorities? Rerun with -O tg or -O pp."

  TLIST=$(pick_threads_list)
  bench threads -ngl 99 -ncmoe "$NC" -fa "$FA" -b "$B" -ub "$UB" $MMAP_FLAG \
        -t "$TLIST" -p 0 -n 96 || true
  T=$(pick_best_threads "$OUTDIR/raw_threads.jsonl")
  log "threads decision: $T"

  if [ $QUICK -eq 0 ]; then
    bench mmap -ngl 99 -ncmoe "$NC" -fa "$FA" -b "$B" -ub "$UB" -t "$T" \
          -mmp 0,1 -p 2048 -n 96 || true
    MM=$(pick_mmap "$OUTDIR/raw_mmap.jsonl")
    [ "$MM" = "1" ] && { MMAP_FLAG=""; SRV_MMAP=""; }
    log "mmap decision: $([ "${MM:-0}" = "0" ] && echo 'no-mmap' || echo 'mmap on')"
    bench kvq8_report -ngl 99 -ncmoe "$NC" -fa "$FA_ON" -b "$B" -ub "$UB" $MMAP_FLAG -t "$T" \
          -ctk q8_0 -ctv q8_0 -p 512 -n 96 -d "$D_MED" || true
  fi
  ;;

# ----------------------------- dense + GPU ----------------------------------
dense)
  log "=== dense path: full-offload fit -> partial -ngl fit (if needed) -> fa -> ub ==="
  NGL_CEILING=""; PARTIAL_NOTE=""
  if probe_server -ngl 99 -fa "$FA_ON" -t "$PHYS"; then
    log "ctx=$CTX fits fully offloaded with f16 KV."
  elif probe_server -ngl 99 -fa "$FA_ON" -ctk q8_0 -ctv q8_0 -t "$PHYS"; then
    SRV_KV="-ctk q8_0 -ctv q8_0"; KV_ARGS="-ctk q8_0 -ctv q8_0"
    log "ctx=$CTX fits fully offloaded only with symmetric q8_0 KV."
  else
    # Model is too large to fully fit in VRAM at this context. Search
    # PARTIAL GPU offload (-ngl < 99) at the ORIGINAL target ctx before ever
    # reducing context — same philosophy as the MoE path's -ncmoe search:
    # negotiate the GPU/CPU split, don't sacrifice context first.
    log "Model does not fit fully offloaded at ctx=$CTX (even with q8_0 KV). Searching partial GPU offload (-ngl) instead of shrinking context."
    for kv_try in "" "-ctk q8_0 -ctv q8_0"; do
      ceiling=$(ngl_ceiling -fa "$FA_ON" $kv_try -t "$PHYS")
      if [ "$ceiling" != "none" ]; then
        NGL_CEILING=$ceiling
        NGL=$(( ceiling - MARGIN )); [ $NGL -lt 0 ] && NGL=0
        SRV_KV="$kv_try"; KV_ARGS="$kv_try"
        log "partial offload: ceiling=$ceiling -> using ngl=$NGL (margin -$MARGIN) kv=${kv_try:-f16}"
        break
      fi
    done
    if [ -z "$NGL_CEILING" ]; then
      # Last resort: even ngl=0 (fully CPU/RAM) doesn't fit at this ctx —
      # a genuine RAM-capacity problem, not a GPU one. Shrink ctx and retry
      # the partial-offload search at each smaller size.
      FIT=0
      while [ "$CTX" -gt 2048 ]; do
        CTX=$(( CTX / 2 ))
        ceiling=$(ngl_ceiling -fa "$FA_ON" -ctk q8_0 -ctv q8_0 -t "$PHYS")
        if [ "$ceiling" != "none" ]; then
          NGL_CEILING=$ceiling
          NGL=$(( ceiling - MARGIN )); [ $NGL -lt 0 ] && NGL=0
          SRV_KV="-ctk q8_0 -ctv q8_0"; KV_ARGS="-ctk q8_0 -ctv q8_0"; FIT=1
          log "reduced to max fitting ctx=$CTX via partial offload ngl=$NGL."; break
        fi
      done
      [ "${FIT:-0}" -eq 1 ] || { log "FATAL: model does not fit even fully on CPU at ctx>=2048 — exceeds available RAM. Use a smaller quant."; exit 2; }
      D_MED=$(( CTX / 4 )); [ $D_MED -lt 1024 ] && D_MED=1024
    fi
    PARTIAL_NOTE="NOTE: this model does not fully fit in VRAM at ctx=$CTX. Using partial GPU offload (ngl=$NGL of 99, ceiling observed at $NGL_CEILING) — layers beyond $NGL run on CPU, which will be substantially slower per-token than a model that fits fully on GPU. This is expected for a model this size on this card, not a misconfiguration."
    log "$PARTIAL_NOTE"
  fi

  FA=$(decide_fa -ngl "$NGL" $KV_ARGS)
  log "flash-attn decision: $FA"

  bench ub_dense -ngl "$NGL" -fa "$FA" $KV_ARGS -b 2048 -ub 512,1024 -p 2048 -n 0 || true
  UP=$(python3 - "$OUTDIR/raw_ub_dense.jsonl" <<'PY'
import json,sys
d={}
try:
    for line in open(sys.argv[1]):
        r=json.loads(line)
        if r.get("n_gen",0)==0: d[r["n_ubatch"]]=r["avg_ts"]
except Exception: pass
print(1024 if d.get(1024,0) > d.get(512,0)*1.03 else 512)
PY
)
  if [ "$UP" = "1024" ] && probe_server -ngl "$NGL" -fa "$FA" $SRV_KV -b 2048 -ub 1024 -t "$PHYS"; then UB=1024; fi
  log "ub decision: $UB"

  # Threads matter here too, unlike full-offload dense: if NGL < 99, real
  # compute is happening on CPU for the un-offloaded layers.
  if [ "$NGL" -lt 99 ]; then
    TLIST=$(pick_threads_list)
    bench threads -ngl "$NGL" -fa "$FA" $KV_ARGS -b 2048 -ub "$UB" -t "$TLIST" -p 0 -n 96 || true
    T=$(pick_best_threads "$OUTDIR/raw_threads.jsonl")
    log "threads decision: $T (partial offload — threads are not empirically flat here)"
  else
    T=$PHYS
  fi
  ;;

# ------------------------------- CPU only -----------------------------------
cpu)
  log "=== CPU-only path: RAM fit -> threads (primary lever) -> ub -> mmap ==="
  probe_server -t "$PHYS" || { log "FATAL: model+ctx does not fit in RAM at ctx=$CTX. Reduce -c or quant."; exit 2; }

  TLIST=$(pick_threads_list)
  bench threads -t "$TLIST" -p 0 -n 64 || true
  T=$(pick_best_threads "$OUTDIR/raw_threads.jsonl")
  log "threads decision: $T"

  bench ub_cpu -t "$T" -b 2048 -ub 256,512 -p 1024 -n 0 || true
  UP=$(python3 - "$OUTDIR/raw_ub_cpu.jsonl" <<'PY'
import json,sys
d={}
try:
    for line in open(sys.argv[1]):
        r=json.loads(line)
        if r.get("n_gen",0)==0: d[r["n_ubatch"]]=r["avg_ts"]
except Exception: pass
print(max(d, key=d.get) if d else 512)
PY
)
  UB=${UP:-512}
  if [ $QUICK -eq 0 ]; then
    bench mmap -t "$T" -ub "$UB" -mmp 0,1 -p 1024 -n 64 || true
    MM=$(pick_mmap "$OUTDIR/raw_mmap.jsonl")
    [ "$MM" = "1" ] && { MMAP_FLAG=""; SRV_MMAP=""; }
    log "mmap decision: $([ "${MM:-0}" = "0" ] && echo 'no-mmap' || echo 'mmap on')"
  fi
  ;;
esac

# ---------------------- stability validation --------------------------------
# The final config must survive $STAB_REPS consecutive full-ctx server loads,
# each including a real decode. On any failure (MoE only), the margin is
# widened by 1 and the FULL rep count is re-run — a single passing retry is
# not sufficient evidence at a stochastic boundary.
log "=== stability: ${STAB_REPS}x restart at full ctx with real decode ==="
STABLE=no
# The loop bound (4) caps how many times we widen the safety margin before
# giving up; the counter itself is intentionally unused (each pass either
# succeeds and breaks, or widens NC/NGL and retries).
for _ in 1 2 3 4; do
  assemble_final
  ok=yes
  for i in $(seq 1 "$STAB_REPS"); do
    probe_server "${FINAL_FLAGS[@]}" || { ok=no; break; }
  done
  if [ "$ok" = "yes" ]; then STABLE=yes; break; fi
  if [ "$MODE" = "moe" ] && [ "${NC:-99}" -lt 99 ]; then
    NC=$(( NC + 1 ))
    log "stability failure at boundary — widening to ncmoe=$NC and re-running full ${STAB_REPS}x validation"
  elif [ "$MODE" = "dense" ] && [ "${NGL:-99}" -lt 99 ] && [ "${NGL:-0}" -gt 0 ]; then
    NGL=$(( NGL - 1 ))
    log "stability failure at partial-offload boundary — reducing to ngl=$NGL and re-running full ${STAB_REPS}x validation"
  else
    log "stability failure and no safe knob to widen in mode=$MODE — reduce -c or quant."
    break
  fi
done
assemble_final

# ---------------------- final confirmation bench ----------------------------
# One authoritative measurement of the exact final config, so the summary
# reports numbers, not just decisions.
# MMAP_FLAG / KV_ARGS deliberately hold multi-token strings (e.g. "-ctk q8_0
# -ctv q8_0") that must word-split into separate argv entries. read -ra makes
# that split explicit and shellcheck-clean (vs. relying on unquoted expansion).
FINAL_BENCH_ARGS=()
_extra=()
case "$MODE" in
  moe)   FINAL_BENCH_ARGS=(-ngl 99 -ncmoe "$NC" -fa "$FA" -b "$B" -ub "$UB" -t "$T"); read -ra _extra <<< "$MMAP_FLAG" ;;
  dense) FINAL_BENCH_ARGS=(-ngl "$NGL" -fa "$FA" -b 2048 -ub "$UB"); read -ra _extra <<< "$KV_ARGS" ;;
  cpu)   FINAL_BENCH_ARGS=(-t "$T" -b 2048 -ub "$UB"); read -ra _extra <<< "$MMAP_FLAG" ;;
esac
[ ${#_extra[@]} -gt 0 ] && FINAL_BENCH_ARGS+=("${_extra[@]}")
bench final "${FINAL_BENCH_ARGS[@]}" -p 512 -n 96 -d "$D_MED" || true
FINAL_TG=$(get_ts "$OUTDIR/raw_final.jsonl" tg)
FINAL_PP=$(get_ts "$OUTDIR/raw_final.jsonl" pp)

if [ $QUICK -eq 0 ]; then
  D2=$(( CTX / 2 )); [ $D2 -gt 8192 ] && D2=8192
  bench depth_curve "${FINAL_BENCH_ARGS[@]}" -p 512 -n 96 -d "0,$D_MED,$D2" || true
fi

# ------------------------------- outputs ------------------------------------
# NOTE: --port 8081 here is a conventional placeholder for the emitted launch
# command / llama-swap snippet, NOT the port tuning ran on (probing used a
# dynamically-found free port). If 8081 is occupied on your machine, change it
# in the final command — it has no bearing on the tuning results themselves.
FINAL_CMD="$SERVER -m $MODEL ${FINAL_FLAGS[*]} --host 127.0.0.1 --port 8081"
DEPTH_LINE=""
[ -s "$OUTDIR/raw_depth_curve.jsonl" ] && DEPTH_LINE=$(jsonl_summary "$OUTDIR/raw_depth_curve.jsonl")
BOUNDARY_NOTE=""
if [ "$MODE" = "moe" ] && [ -n "$OBSERVED_FLOOR" ]; then
  GAP=$(( NC - OBSERVED_FLOOR ))
  if [ "$GAP" -le 1 ]; then
    BOUNDARY_NOTE="NOTE: final ncmoe ($NC) is adjacent to the observed fail boundary ($OBSERVED_FLOOR). This boundary is stochastic (allocator variance) — the same value can pass one run and fail another. If you ever see load failures in production, bump ncmoe by +1, or rerun with -M 2 to bake in extra margin."
  fi
elif [ "$MODE" = "dense" ] && [ -n "$NGL_CEILING" ]; then
  GAP=$(( NGL_CEILING - NGL ))
  if [ "$GAP" -le 1 ]; then
    BOUNDARY_NOTE="NOTE: final ngl ($NGL) is adjacent to the observed fail ceiling ($NGL_CEILING). This boundary is stochastic (allocator variance) — the same value can pass one run and fail another. If you ever see load failures in production, reduce ngl by 1, or rerun with -M 2 to bake in extra margin."
  fi
  [ -n "$PARTIAL_NOTE" ] && BOUNDARY_NOTE="$PARTIAL_NOTE $BOUNDARY_NOTE"
fi

{
  echo "# Autotune summary — $MODELNAME"
  echo ""
  echo "- Date: $(date) | mode: $MODE | Stable: $STABLE (${STAB_REPS}x restart-validated)"
  echo "- Host: $DISTRO_ID | CPU: $PHYS phys/$LOGICAL logical | GPU: ${GPU_NAME:-none} (${VRAM_TOTAL:-0} MiB)"
  echo "- Target ctx: $CTX (validated by real server load + decode)"
  echo "- Objective: $OBJECTIVE | ncmoe margin: +$MARGIN"
  echo "- Decisions: fa=$FA ub=$UB b=${B:-2048}$([ "$MODE" = "moe" ] && echo " ncmoe=$NC (observed floor: ${OBSERVED_FLOOR:-n/a})")$([ "$MODE" = "dense" ] && [ "$NGL" -lt 99 ] && echo " ngl=$NGL/99 (partial offload, ceiling: ${NGL_CEILING:-n/a})") t=$T mmap=$([ -n "${SRV_MMAP:-}" ] && echo off || echo on) kv=${SRV_KV:-f16} arch=$ARCH"
  [ -n "$BOUNDARY_NOTE" ] && { echo ""; echo "> $BOUNDARY_NOTE"; }
  echo ""
  echo "## Measured performance (final config)"
  echo "- tg128 @ d=$D_MED: **${FINAL_TG:-n/a} t/s**   |   pp512 @ d=$D_MED: **${FINAL_PP:-n/a} t/s**"
  [ -n "$DEPTH_LINE" ] && echo "- Depth curve: $DEPTH_LINE"
  echo ""
  if [ "$MODE" = "moe" ] && [ -n "$CAND_TABLE" ]; then
    echo "## ub candidates (trade-off table, objective=$OBJECTIVE)"
    echo ""
    echo "| ub | observed floor | ncmoe used | tg (t/s) | pp (t/s) | score |"
    echo "|----|----------------|------------|----------|----------|-------|"
    printf '%s' "$CAND_TABLE"
    echo ""
    echo "Different priority? Rerun with \`-O tg\` (max generation) or \`-O pp\` (max prompt ingestion)."
    echo ""
  fi
  echo "## Final command"
  echo '```bash'
  echo "$FINAL_CMD"
  echo '```'
  echo ""
  echo "## Model-specific flags (derived, not guessed)"
  echo "- Architecture (from GGUF metadata): \`$ARCH\`${MODEL_DISPLAY_NAME:+ ($MODEL_DISPLAY_NAME)}"
  if [ -n "$JINJA_FLAG" ]; then
    echo "- \`--jinja\`: **added**. This file embeds its own chat template (tokenizer.chat_template"
    echo "  present in GGUF metadata). llama.cpp's built-in templates are a fixed, older set that"
    echo "  can silently mis-render newer or custom templates — using the model's own template is"
    echo "  strictly safer whenever one exists."
  else
    echo "- \`--jinja\`: not added. No embedded chat template was found in this GGUF; llama.cpp will"
    echo "  use its built-in template matching (or --chat-template if you set one manually)."
  fi
  if [ -n "$SAMPLING_FLAGS" ]; then
    echo "- Sampling: **\`$SAMPLING_FLAGS\`** — $SAMPLING_NOTE"
  else
    echo "- Sampling: no documented quirk for architecture \`$ARCH\` in this script's table;"
    echo "  llama.cpp defaults were left as-is. Check the model card if output quality looks"
    echo "  off — this table only covers architectures with a *known, documented* reason to"
    echo "  deviate. If you find one, add it to the SAMPLING_FLAGS case statement and share it."
  fi
  echo ""
  echo "## llama-swap snippet"
  echo '```yaml'
  echo "  \"$MODELNAME\":"
  echo "    ttl: 600"
  echo "    cmd: |"
  echo "      $SERVER"
  echo "      --port \${PORT}"
  echo "      -m $MODEL"
  echo "      ${FINAL_FLAGS[*]}"
  echo '```'
  echo ""
  echo "Raw per-phase results: raw_*.jsonl. Probe log: probe_server.log. Host info: meta.txt. Machine-readable: SUMMARY.json"
} > "$SUMMARY"

# machine-readable results (for aggregating runs across machines)
export S_MODE="$MODE" S_STABLE="$STABLE" S_CTX="$CTX" S_FA="$FA" S_UB="$UB" S_B="${B:-2048}"
export S_NC="${NC:-}" S_T="$T" S_ARCH="$ARCH" S_MODELNAME="$MODELNAME" S_MODEL="$MODEL"
export S_OBJECTIVE="$OBJECTIVE" S_MARGIN="$MARGIN" S_STAB_REPS="$STAB_REPS"
export S_GPU="${GPU_NAME:-none}" S_VRAM="${VRAM_TOTAL:-0}" S_FLOOR="${OBSERVED_FLOOR:-}"
export S_NGL="${NGL:-99}" S_NGL_CEILING="${NGL_CEILING:-}"
export S_TG="${FINAL_TG:-}" S_PP="${FINAL_PP:-}" S_KV="${SRV_KV:-f16}" S_CMD="$FINAL_CMD"
S_MMAP_VAL="$([ -n "${SRV_MMAP:-}" ] && echo off || echo on)"
export S_MMAP="$S_MMAP_VAL" S_DISTRO="$DISTRO_ID"
python3 - > "$OUTDIR/SUMMARY.json" <<'PY'
import json, os
g = os.environ.get
out = {
  "model": g("S_MODELNAME"), "model_path": g("S_MODEL"), "arch": g("S_ARCH"),
  "mode": g("S_MODE"), "stable": g("S_STABLE") == "yes",
  "host": {"distro": g("S_DISTRO"), "gpu": g("S_GPU"), "vram_mib": int(g("S_VRAM") or 0)},
  "target_ctx": int(g("S_CTX") or 0),
  "objective": g("S_OBJECTIVE"),
  "decisions": {
    "flash_attn": g("S_FA"), "ubatch": int(g("S_UB") or 0), "batch": int(g("S_B") or 0),
    "ncmoe": int(g("S_NC")) if g("S_NC") else None,
    "ncmoe_observed_floor": int(g("S_FLOOR")) if g("S_FLOOR") else None,
    "ncmoe_margin": int(g("S_MARGIN") or 0),
    "ngl": int(g("S_NGL") or 99),
    "ngl_observed_ceiling": int(g("S_NGL_CEILING")) if g("S_NGL_CEILING") else None,
    "threads": int(g("S_T") or 0), "mmap": g("S_MMAP"), "kv_cache": g("S_KV"),
  },
  "measured": {
    "tg_ts": float(g("S_TG")) if g("S_TG") else None,
    "pp_ts": float(g("S_PP")) if g("S_PP") else None,
  },
  "stability_reps": int(g("S_STAB_REPS") or 0),
  "final_command": g("S_CMD"),
}
print(json.dumps(out, indent=2))
PY

log "============================================================"
log "DONE. Mode: $MODE | Stable: $STABLE | tg=${FINAL_TG:-n/a} pp=${FINAL_PP:-n/a}"
[ -n "$BOUNDARY_NOTE" ] && log "$BOUNDARY_NOTE"
log "Final: $FINAL_CMD"
log "Summary: $SUMMARY (+ SUMMARY.json)"

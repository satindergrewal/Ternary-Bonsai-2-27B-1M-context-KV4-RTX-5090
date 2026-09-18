#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Satinder Grewal
# ============================================================================
# start.sh - Bonsai 2 27B (ternary hybrid) at 1M context on ONE RTX 5090 (32GB).
#
# Serves the PrismML qwen35-architecture ternary checkpoints through the
# Bonsai-demo llama-server. ABLIT=1 in .env switches to the abliterated
# BoldingBuilds PTQ1_0 checkpoint; ABLIT=0 serves the base PrismML PQ2_0.
#
# ---------------------------------------------------------------------------
# HOW IT FITS (measured 2026-09-18 on this box - RTX 5090, 32.6 GiB)
#
#   weights .................  5.94 GiB   (PTQ1_0 1.75bpw) or 7.21 (PQ2_0 2.13)
#   KV cache (q4_0, 1M) ..... 17.92 GiB   (18 KiB/token: only 16 of 64 blocks
#                                          carry attention KV; the other 48 are
#                                          GDN/SSM with constant state)
#   mmproj + buffers/graphs .   ~6-8 GiB  (grows a little with vision prompts)
#   idle total .............. 30.7 GiB   (PTQ1_0) / 32.0 GiB (PQ2_0)
#
# fp16 KV at 1M would be 64 GiB - KV4 is what makes this possible. The other
# half of the trick is patches/patch-gguf-context.py: llama-server caps slot
# context at the model's n_ctx_train, so an unpatched GGUF silently turns
# -c 1048576 into 262K lanes (boot log: "capping").
#
# Decode is bandwidth-friendly at shallow depth (~132 t/s) and falls with
# depth as the attention scan grows (16.9 t/s @247K, ~5 t/s near 1M).
# See docs/BENCH.md for every number.
#
# USAGE:
#   ./start.sh              start using .env (copies .env.sample defaults if
#                           .env is absent)
#   ./start.sh CHECK        print resolved config, touch nothing
#   CONTEXT=262144 ./start.sh   any knob can be overridden per invocation
#   ./stop.sh               stop
# ============================================================================

set -u
cd "$(dirname "$0")"

# ---- config: .env, else .env.sample, else baked defaults -------------------
# Caller exports win over the file, including explicit empties (ABLIT= ./start.sh).
load_env() {
  [ -f "$1" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in *[!A-Za-z0-9_]*|"") continue ;; esac
    v="${v%\"}"; v="${v#\"}"
    if [ -z "${!k+x}" ]; then export "$k=$v"; fi
  done < "$1"
}
if [ -f .env ]; then
  load_env .env
elif [ -f .env.sample ]; then
  echo "(no .env found - using .env.sample defaults; cp .env.sample .env to customize)"
  load_env .env.sample
fi
ABLIT="${ABLIT:-1}"
CONTEXT="${CONTEXT:-1048576}"
KV4="${KV4:-1}"
DFLASH2="${DFLASH2:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8013}"
PARALLEL="${PARALLEL:-1}"
DEMO="${DEMO:-$HOME/Bonsai-demo}"

if [ "${ABLIT}" = "1" ]; then
  GGUF="${MODEL:-$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf}"
  DEF_ALIAS="Bonsai-2-27B-Ablit"
else
  GGUF="${MODEL:-$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
  DEF_ALIAS="Bonsai-2-27B"
fi
ALIAS="${ALIAS:-$DEF_ALIAS}"

# DFlash2 needs ~2 GiB for the drafter; clamp context so both fit a 32GB card.
if [ "$DFLASH2" = "1" ] && [ "$CONTEXT" -gt 921600 ]; then
  CONTEXT=921600
fi

if [ "${1:-}" = "CHECK" ]; then
  SCALE=$(awk "BEGIN{if ($CONTEXT > 262144) printf \"%.4f\", $CONTEXT/262144; else printf \"1\"}")
  echo "model:      $GGUF"
  echo "alias:      $ALIAS"
  echo "context:    $CONTEXT (yarn scale $SCALE)"
  echo "kv4: $KV4  dflash2: $DFLASH2  parallel: $PARALLEL"
  echo "bind:       $HOST:$PORT   demo: $DEMO"
  exit 0
fi

if curl -s -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; then
  echo "ALREADY-RUNNING (./stop.sh first to change config)"
  exit 0
fi
[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF (run ./download.sh or fix MODEL)"; exit 2; }

EXTRA+=(-m "$GGUF")
YARN=()
if [ "$CONTEXT" -gt 262144 ]; then
  SCALE=$(awk "BEGIN{printf \"%.4f\", $CONTEXT/262144}")
  YARN+=(--yarn-orig-ctx 262144 --rope-scale "$SCALE")
fi
DRAFT=()
if [ "$DFLASH2" = "1" ]; then
  DRAFT+=(--spec-type draft-dflash --model-draft "$DEMO/models/dflash2/Bonsai-2-27B-DFlash2-Q8_0.gguf" --spec-draft-n-max 5 -ngld 999)
fi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export BONSAI_KV4="$KV4" BONSAI_CTX="$CONTEXT" BONSAI_HOST="$HOST" PORT="$PORT"

cd "$DEMO"
setsid nohup ./scripts/start_llama_server.sh \
  ${YARN[@]+"${YARN[@]}"} --parallel "$PARALLEL" \
  --alias "$ALIAS" ${EXTRA[@]+"${EXTRA[@]}"} ${DRAFT[@]+"${DRAFT[@]}"} \
  > "$HOME/bonsai-serve.log" 2>&1 < /dev/null &
echo "launching ablit=$ABLIT ctx=$CONTEXT draft=$([ "$DFLASH2" = "1" ] && echo on || echo off) (log: ~/bonsai-serve.log) ..."
for i in $(seq 1 30); do
  if curl -s -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "READY after ~${i}0s"
    PORT="$PORT" python3 - <<'PYINFO'
import json, os, urllib.request
port = os.environ.get("PORT", "8013")
m = json.load(urllib.request.urlopen("http://localhost:%s/v1/models" % port, timeout=5))["data"][0]
print("model id:", m["id"])
print("ctx:", m["meta"]["n_ctx"], "| quant:", m["meta"]["ftype"])
PYINFO
    exit 0
  fi
  if ! pgrep -f "start_llama_server.sh|bin/cuda/llama-server" >/dev/null 2>&1; then
    echo "BOOT-DIED:"
    grep -aE "error|Error|failed" "$HOME/bonsai-serve.log" | tail -3
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT - log tail:"
tail -n 12 "$HOME/bonsai-serve.log"
exit 1

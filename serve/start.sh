#!/bin/bash
# Bonsai-2 27B serve wrapper. Config via serve.env (see serve.env.example).
# ./start.sh            start (no-op if already running)
# ./start.sh CHECK      print resolved config and exit
# ./stop.sh             stop
set -u
cd "$(dirname "$0")"

# ---- config ----
if [ -f serve.env ]; then set -a; . ./serve.env; set +a; fi
ABLIT="${ABLIT:-1}"
CONTEXT="${CONTEXT:-1048576}"
KV4="${KV4:-1}"
DFLASH2="${DFLASH2:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8013}"
PARALLEL="${PARALLEL:-1}"
DEMO="${DEMO:-$HOME/Bonsai-demo}"

if [ -n "${MODEL:-}" ]; then
  GGUF="$MODEL"
elif [ "$ABLIT" = "1" ]; then
  GGUF="$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf"
else
  GGUF="$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
fi
if [ -n "${ALIAS:-}" ]; then ALIAS="${ALIAS}"; elif [ "$ABLIT" = "1" ]; then ALIAS="Bonsai-2-27B-Ablit"; else ALIAS="Bonsai-2-27B"; fi

# DFlash2 is opt-in and measured net-negative on a 5090 (docs/BENCH.md);
# the drafter needs ~2 GiB, so clamp context when it is enabled.
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
[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF"; exit 2; }

EXTRA=()
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
echo "launching ctx=$CONTEXT model=$([ "$ABLIT" = "1" ] && echo ablit || echo base) draft=$([ "$DFLASH2" = "1" ] && echo on || echo off) (log: ~/bonsai-serve.log) ..."
for i in $(seq 1 30); do
  if curl -s -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "READY after ~${i}0s"
    python3 "$(dirname "$0")/model-info.py"
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

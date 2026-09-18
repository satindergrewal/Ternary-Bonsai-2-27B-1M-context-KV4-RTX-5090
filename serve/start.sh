#!/bin/bash
# Bonsai-2 27B serve wrapper - picks a variant like a recipe file.
#   ./start.sh               default variant (1m-ablit)
#   ./start.sh <variant>     e.g. 1m-base, 900k-dflash2, 262k-base-4lane
#   ./start.sh CHECK [v]     print resolved config, touch nothing
#   ./stop.sh                stop the server
# Variants live in serve/variants/<name>.env (copy one to make your own).
set -u
cd "$(dirname "$0")"

VARIANT="${1:-1m-ablit}"
if [ "$VARIANT" = "CHECK" ]; then
  VARIANT="${2:-1m-ablit}"; MODE=check
fi
VFILE="variants/$VARIANT.env"
[ -f "$VFILE" ] || { echo "unknown variant: $VARIANT (have: $(ls variants/ | sed 's/\.env$//' | tr '\n' ' '))"; exit 2; }
set -a; . "./$VFILE"; set +a

CONTEXT="${CONTEXT:-1048576}"
KV4="${KV4:-1}"
DFLASH2="${DFLASH2:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8013}"
PARALLEL="${PARALLEL:-1}"
DEMO="${DEMO:-$HOME/Bonsai-demo}"

if [ "${ABLIT:-1}" = "1" ]; then
  GGUF="${MODEL:-$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf}"
  DEF_ALIAS="Bonsai-2-27B-Ablit"
else
  GGUF="${MODEL:-$DEMO/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
  DEF_ALIAS="Bonsai-2-27B"
fi
ALIAS="${ALIAS:-$DEF_ALIAS}"

if [ "$DFLASH2" = "1" ] && [ "$CONTEXT" -gt 921600 ]; then
  CONTEXT=921600
fi

if [ "${MODE:-}" = "check" ]; then
  SCALE=$(awk "BEGIN{if ($CONTEXT > 262144) printf \"%.4f\", $CONTEXT/262144; else printf \"1\"}")
  echo "variant:    $VARIANT"
  echo "model:      $GGUF"
  echo "alias:      $ALIAS"
  echo "context:    $CONTEXT (yarn scale $SCALE)"
  echo "kv4: $KV4  dflash2: $DFLASH2  parallel: $PARALLEL"
  echo "bind:       $HOST:$PORT   demo: $DEMO"
  exit 0
fi

if curl -s -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; then
  echo "ALREADY-RUNNING (./stop.sh first to change variant)"
  exit 0
fi
[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF"; exit 2; }

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
echo "launching variant=$VARIANT ctx=$CONTEXT draft=$([ "$DFLASH2" = "1" ] && echo on || echo off) (log: ~/bonsai-serve.log) ..."
for i in $(seq 1 30); do
  if curl -s -m 3 "localhost:$PORT/health" 2>/dev/null | grep -q ok; then
    echo "READY after ~${i}0s"
    PORT="$PORT" python3 "$(dirname "$0")/model-info.py"
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

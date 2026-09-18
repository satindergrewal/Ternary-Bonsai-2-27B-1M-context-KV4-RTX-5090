#!/bin/bash
# Bonsai-2 27B @ 1M ctx, KV4, single lane on :8013
# Optional env: BONSAI_GGUF=/abs/path.gguf, BONSAI_DISPLAY="name", BONSAI_CTX=N, DFLASH2=1
# DFLASH2 is OPT-IN and measured net-NEGATIVE in this build (acceptance ~0.11-0.22
# vs 0.40 on the author runtime); it auto-clamps ctx to 900K. Default = no draft.
set -u
cd "$(dirname "$0")"

GGUF="${BONSAI_GGUF:-$HOME/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf}"
CTX="${BONSAI_CTX:-1048576}"
ALIAS="${BONSAI_DISPLAY:-Bonsai-2-27B-Ablit}"

if curl -s -m 3 localhost:8013/health 2>/dev/null | grep -q ok; then
  echo "ALREADY-RUNNING (stop first to change ctx/model)"
  exit 0
fi
[ -f "$GGUF" ] || { echo "GGUF not found: $GGUF"; exit 2; }

EXTRA+=(-m "$GGUF")
DRAFT=()
if [ "${DFLASH2:-0}" = "1" ]; then
  [ "$CTX" -gt 921600 ] && CTX=921600
  DRAFT+=(--spec-type draft-dflash --model-draft "$HOME/Bonsai-demo/models/dflash2/Bonsai-2-27B-DFlash2-Q8_0.gguf" --spec-draft-n-max 5 -ngld 999)
fi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export BONSAI_KV4=1 BONSAI_CTX="$CTX" BONSAI_HOST=0.0.0.0 PORT=8013
setsid nohup ./scripts/start_llama_server.sh \
  --yarn-orig-ctx 262144 --rope-scale 4 --parallel 1 \
  --alias "$ALIAS" ${EXTRA[@]+"${EXTRA[@]}"} ${DRAFT[@]+"${DRAFT[@]}"} \
  > "$HOME/bonsai-serve-1m.log" 2>&1 < /dev/null &
echo "launching ctx=$CTX draft=$([ ${#DRAFT[@]} -gt 0 ] && echo on || echo off) ..."
for i in $(seq 1 30); do
  if curl -s -m 3 localhost:8013/health 2>/dev/null | grep -q ok; then
    echo "READY after ~${i}0s"
    python3 "$(dirname "$0")/model-info.py"
    exit 0
  fi
  if ! pgrep -f "start_llama_server.sh|bin/cuda/llama-server" >/dev/null 2>&1; then
    echo "BOOT-DIED:"
    grep -aE "error|Error|failed" "$HOME/bonsai-serve-1m.log" | tail -3
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT - log tail:"
tail -n 12 "$HOME/bonsai-serve-1m.log"
exit 1

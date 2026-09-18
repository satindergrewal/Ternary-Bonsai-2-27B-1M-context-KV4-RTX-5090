#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Satinder Grewal
# Download the target GGUF into the demo repo and apply the 1M context patch.
#   ABLIT=1 ./download.sh   (default) abliterated PTQ1_0, BoldingBuilds
#   ABLIT=0 ./download.sh   base PQ2_0, PrismML
# Both land in $DEMO/models/bonsai2-gguf/27B/. Re-run make_kv_bias.sh in the
# demo repo after switching quants (the KV bias is quant-specific).
set -eu
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
ABLIT="${ABLIT:-1}"
DEMO="${DEMO:-$HOME/Bonsai-demo}"
DIR="$DEMO/models/bonsai2-gguf/27B"
mkdir -p "$DIR"

if [ "$ABLIT" = "1" ]; then
  REPO="BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF"
  FILE="Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf"
else
  REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
  FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf"
fi
echo "downloading $REPO/$FILE ..."
hf download "$REPO" "$FILE" --local-dir "$DIR"
python3 patches/patch-gguf-context.py "$DIR/$FILE" 1048576
echo "done. if you switched quants, rebuild the KV bias:"
echo "  cd $DEMO && ./scripts/make_kv_bias.sh"

# Changelog

## 2026-09-18

- Initial release: 1M-context serving recipe for Bonsai 2 27B on a single
  32GB GPU (RTX 5090).
- `patches/patch-gguf-context.py`: 4-byte GGUF header patch lifting the
  llama-server n_ctx_train slot cap (262144 -> 1048576). Weights untouched.
- `.env.sample` + `start.sh`/`stop.sh`/`download.sh`: env-driven serve with
  ABLIT / CONTEXT / KV4 / DFLASH2 knobs, derived yarn scale and alias.
- Measured receipts: prefill 934-1124 t/s at 250-315K depth, decode 132 t/s
  shallow falling to ~14 t/s at 315K, VRAM 30.7-32.0 of 32.6 GiB, 314K-token
  single-request proof.
- DFlash2 re-measured and parked: acceptance 0.11-0.22 here vs 0.400 on the
  author's RTX PRO 6000; net-negative on this GPU. Full matrix in
  docs/BENCH.md.

# Ternary Bonsai 2 27B at 1M context, q4 KV cache - single RTX 5090 (32GB)

A serving recipe for the ternary hybrid Bonsai 2 27B (PQ2_0 or PTQ1_0) with a
true 1M-token per-request window, a q4_0 KV cache with mean-centering bias,
and vision, on one 32GB card. This is a recipe, not a model release: the model
files are the official PrismML GGUFs with 4 bytes of header metadata patched
locally. No weights are modified anywhere in this repo.

## Testing status: spot-checked, not rigorously evaluated

Everything below was measured on one box on 2026-09-18: boots, endpoint
probes, a 314K-token single-request comprehension check (coherent, correct
answer), and prefill/decode timings from the server log. No long-context
evaluation suite has been run. Low-bit ternary models are known to loop at
depth (see the Qwen 350K and GLM 600K precedents), so treat very long
generations with the usual care. The region past 262K rides YaRN x4
extrapolation; quality at depth beyond a coherence check is unverified.

## Why a 27B fits 1M in 32GB

Bonsai 2 27B is a qwen35-family hybrid: 64 blocks, but only 16 carry a KV
cache (4 kv heads x 256 head dim); the other 48 are linear-attention/SSM
blocks with constant state. That makes the cache tiny by dense-model standards:

| KV dtype | bytes per token | 1M tokens |
|---|---:|---:|
| fp16 | 64 KiB | 64 GiB (does not fit) |
| q4_0 | 18 KiB | 17.9 GiB (fits) |

Weights at 1.75-2.13 bpw add 5.9-7.2 GiB, so a 1M window plus weights plus
buffers lands at 30.7-32.0 GiB. Tight but real. Vision adds activations on
top; if you ever OOM on image prompts, drop the context a notch.

## The one patch you need

llama-server caps slot context at the model's trained context (n_ctx_train,
derived from the GGUF context_length key). Bonsai 2 ships at 262144, and a
bigger `-c` gets silently re-capped per slot ("the slot context exceeds the
training context - capping"). The fix is 4 bytes:

```bash
python3 patches/patch-gguf-context.py MODEL.gguf 1048576
python3 patches/patch-gguf-context.py MODEL.gguf --check   # verify
```

Weights are untouched. To revert, re-run with 262144. Stop the server before
patching. Keep the rope flags honest afterwards:
`--yarn-orig-ctx 262144 --rope-scale 4`.

## Quick start

1. Get the official demo running: `PrismML-Eng` publishes a `Bonsai-demo`
   repo with prebuilt binaries and `setup.sh`. The prebuilt binary targets
   CUDA 12.8; on a CUDA 13 box either add the cu12 wheels as a shim
   (`uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12`, symlink
   `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` into
   `bin/cuda/`) or build natively:

   ```bash
   git clone -b prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git
   # cmake and ninja must be on PATH (the demo venv has them)
   PATH="$HOME/Bonsai-demo/.venv/bin:$PATH" ./scripts/build_cuda_linux.sh \
       --archs 120 --output cuda-new
   # then stop the server, swap bin/cuda-new over bin/cuda
   ```

   The native build was ~10% faster in our measurements (130.8 vs 119 t/s
   shallow decode) and removes the shim entirely.

2. Patch every GGUF you intend to serve at 1M (see above).

3. Build the KV bias once per model file (it is quant-specific; re-run it
   when you switch quants):

   ```bash
   ./scripts/make_kv_bias.sh
   ```

4. Copy `serve/start.sh`, `serve/stop.sh` and `serve/model-info.py` next to
   the demo repo (or edit the paths) and:

   ```bash
   ./start.sh                    # default: abliterated PTQ1_0, 1M, KV4, alias
   BONSAI_GGUF=/abs/path.gguf BONSAI_DISPLAY="name" ./start.sh
   ./stop.sh
   ```

## Flags that matter

| flag / env | effect |
|---|---|
| `BONSAI_KV4=1` | q4_0 KV cache, auto-loads the mean-centering bias. Without it, 1M KV is 64 GiB and will not fit |
| `BONSAI_CTX=1048576` | the window; requires the patched GGUF |
| `--yarn-orig-ctx 262144 --rope-scale 4` | rope extrapolation past the 262K native context |
| `--parallel 1` | one conversation gets the whole 1M. A second request during a long prefill gets a 503. Drop this for 4 x 262K lanes instead |
| `--alias NAME` | clean model id in the API (llama-server otherwise reports the full file path) |
| `DFLASH2=1` | opt-in speculative decoding, clamps context to 900K. Measured net-negative on this GPU (see below) |

## Results (2026-09-18, RTX 5090)

| measurement | value |
|---|---|
| boot to READY (warm page cache, 1M KV4) | ~20 s |
| VRAM, PTQ1_0 (1.75bpw) at 1M KV4, idle | 30.7 / 32.6 GiB |
| VRAM, PQ2_0 (2.13bpw) at 1M KV4, idle | 32.0 / 32.6 GiB |
| prefill at 247K depth | 1124 t/s |
| prefill at 315K depth | 934 t/s |
| decode, short context (native build) | 130.8 t/s |
| decode vs depth (PQ2_0, KV4) | 119 shallow, 16.9 @247K, 13.7 @314K |
| single request >262K native | 314,666 tokens accepted, coherent answer |

Full numbers and the measurement commands: [docs/BENCH.md](docs/BENCH.md).

## DFlash2 speculative decoding: measured, parked

ProCreations published a DFlash2 drafter adapted to this target. We applied
their patch (it applies cleanly to the prism branch), rebuilt, and measured:

| config | acceptance | decode |
|---|---:|---:|
| card claim (their box, their binary) | 0.400 | 179 t/s blended |
| our build, base PQ2_0, 32K, no yarn | 0.225 | 117.8 t/s |
| our build, 900K, yarn, KV4 | 0.114 | 58.3 t/s |
| their own prebuilt binary on our GPU | 0.198-0.294 | 112-141 t/s |

Ruled out by A/B: sampler settings, KV4 (identical with fp16 KV), quant
mismatch (identical on the base PQ2_0). Long context halves acceptance
further - the drafter never sees scaled rope. Conclusion: the card numbers
are real on their RTX PRO 6000 96GB but do not transfer to a 5090; the
drafter overhead eats the winnings. `DFLASH2=1 ./start.sh` opts back in at
900K if you want to experiment; the default serve runs without it.

Also worth knowing: the slot-context cap and this patch are llama.cpp-level
behavior, so the same 4-byte trick applies to other GGUFs whose native
context you want to exceed with yarn. Their drafter compatibility, however,
is target-specific - a drafter trained for one checkpoint will not speed up
an abliterated or differently-quantized sibling (we measured that too).

## Repo map

| path | contents |
|---|---|
| `patches/patch-gguf-context.py` | the 4-byte context_length patcher (stdlib only) |
| `serve/start.sh`, `serve/stop.sh`, `serve/model-info.py` | tested wrappers: idempotent start, health poll, clean stop |
| `docs/BENCH.md` | full measurements, methodology, DFlash2 matrix |

## Sources and credits

- PrismML-Eng: the llama.cpp fork (ternary quants, DSpark/dflash, KV rotation
  and mean-centering bias), the Bonsai 2 model files, and the Bonsai-demo repo
  (https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). Spec decoding
  rides upstream DSpark work (ggml-org/llama.cpp PR #25173).
- BoldingBuilds: the abliterated PTQ1_0 GGUF this recipe serves by default
  (https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF).
- ProCreations: the DFlash2 drafter, its patch, and the honest benchmark
  reports this page re-measures against
  (https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2).
- Recipe, context patch, and measurements: Satinder Grewal.

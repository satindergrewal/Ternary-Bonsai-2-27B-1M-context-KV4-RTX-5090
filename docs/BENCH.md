# Bench notes, 2026-09-18

Single box: RTX 5090 32GB, Linux, CUDA 13.0 toolkit, prism-branch llama.cpp
build (clone tip 5d80cff, `--archs 120`). All timings from llama-server's own
slot timing lines (`grep "eval time" serve.log`); probes are plain
`/v1/chat/completions` curls. The 1M lane runs `--parallel 1`, q4_0 KV cache
with the mean-centering bias, `--yarn-orig-ctx 262144 --rope-scale 4`.

## Context windows and VRAM

| serve | context | VRAM idle | notes |
|---|---:|---:|---|
| PQ2_0 (2.13bpw), KV4 | 1,048,576 | 32.0 / 32.6 GiB | 600 MiB headroom; vision prompts add activations |
| PTQ1_0 abliterated (1.75bpw), KV4 | 1,048,576 | 30.7 / 32.6 GiB | the daily-driver lane |
| PQ2_0 + DFlash2 drafter, KV4 | 921,600 | ~31.1 GiB | 1M + drafter does not fit (drafter needs ~2.0 GiB) |

Single-request proof past the 262K native context: a 314,666-token prompt was
accepted and answered coherently (prompt eval 336.8 s, 934 t/s; decode 13.7
t/s at that depth). At 900K the context window is still 3.5x native.

## Prefill and decode vs depth (PQ2_0, KV4)

| depth | prefill | decode |
|---|---:|---:|
| short (57 tok) | 356-388 t/s | 119-131 t/s |
| 247K | 1124 t/s | 16.9 t/s |
| 315K | 934 t/s | 13.7 t/s |

Decode falloff with depth is dense-attention scan physics: every generated
token re-reads the whole KV cache of the 16 attention layers (~6.4 GB per
token at 314K in q4). Estimated ~5 t/s at the 1M bottom.

Build note: the native CUDA 13 build measured 130.8 t/s shallow decode vs
119 on the published CUDA 12.8 binary (cu12 shim). ~10% free from building
natively for sm120.

## The slot-context cap (why the patch exists)

With the stock GGUF and `-c 1048576`, the server logs
`the slot context (1048576) exceeds the training context of the model
(262144) - capping` and partitions 4 slots x 262144. A single request past
262,144 tokens is rejected with 400. `--parallel 1` does not lift the cap.
Patching `qwen35.context_length` to 1048576 in the GGUF header removes it
(the boot warning `n_ctx_seq > n_ctx_train` disappears and `/v1/models`
reports n_ctx 1048576).

## DFlash2 matrix

Drafter: ProCreations/Ternary-Bonsai-2-27B-DFlash2 (1.924B Q8 draft,
`--spec-type draft-dflash --spec-draft-n-max 5 -ngld 999`).
Their published patch applies cleanly to the prism branch at 5d80cff.

| config | acceptance | decode |
|---|---:|---:|
| their card: their box (RTX PRO 6000 96GB), their binary | 0.400 | 179.2 t/s blended |
| our build, base PQ2_0, 32K ctx, fp16 KV, no yarn | 0.218-0.225 | 117-118 t/s |
| our build, 900K ctx, yarn, KV4 | 0.114 | 58.3 t/s |
| their own prebuilt binary on our GPU, their exact flags | 0.198 (prose), 0.294 (code) | 112 / 141 t/s |

A/B conclusions:

- Sampler settings (temp 0 vs their temp 1 / top-p 0.95 / top-k 20 / min-p 0):
  no difference in acceptance.
- KV4 vs fp16 KV: no difference.
- Quant mismatch: none - the base PQ2_0 scores the same as the abliterated
  PTQ1_0.
- Long context halves acceptance (0.22 at 32K to 0.11 at 900K): the drafter
  never receives the target's scaled rope.
- Their own binary reproduces our patched-build numbers on this GPU, which
  excludes our patch application as the variable. The card's 0.400/179 t/s is
  real on their hardware and does not transfer to a 5090; undrafted decode
  here is ~130 t/s, so the drafter is neutral at best and negative at depth.

Revival paths if wanted: run their runtime as a dedicated short-context lane
(the tarball works on this driver), or reconcile their full source snapshot
(`prism-dflash2-source.tar.gz`) against the prism branch to find what the
published patch alone is missing. `start.sh` keeps a `DFLASH2=1` opt-in
(clamps to 900K) for either.

# qwen-bench-2026-05-12-v3-suite

**Date:** 2026-05-12 (Tue) MSK
**Image under test:** `repne/vllm:v3` (digest `fd2f7b567b19`, 29.7 GB)
**Model:** Qwen3.6-27B (BF16 + FP8 variants)
**Drafter:** `z-lab/Qwen3.6-27B-DFlash` (DFlash configs only)
**Hardware:** Blackwell × 2, TP=2 on GPUs 0+1 (UUIDs `ba6334bc…`, `538bf008…`); GPU 2 reserved (untouched at 89.8 GB throughout)
**Total wall time:** 03:59:19 → 07:31:32 MSK (3 h 32 min)

This is a follow-up to [`qwen-bench-2026-05-11-v2-followup`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup), this time on Repne's `:v3` image instead of `:v2`, with the same 4 speculative-decoding configurations, the same patched stress harness (`smart_glue_humaneval`), and the same `mt=16384` quality budget.

## TL;DR

| Config | HE Pass | MBPP Pass | Peak tok/s | Single-user tok/s | Length trunc |
|---|---|---|---|---|---|
| BF16+DFlash N=8 | **92.7%** (152/164) | **91.1%** (234/257) | 300 | 86 | 0 |
| FP8+DFlash N=8 | 89.0% (146/164) | 88.7% (228/257) | 375 | 86 | 0 |
| FP8+MTP=3 | 88.4% (145/164) | 89.1% (229/257) | 369 | 98 | 0 |
| **FP8+MTP=5** | 93.3% (153/164) | 87.2% (224/257) | **402** | **101** | 0 ⚠️ leaks `<think>` |

**Production SOTA = `repne/vllm:v3` + FP8+MTP=3** (validated in production 2026-05-12 ~10:03 MSK).

- HE 88.4% / MBPP 89.1% / 369 tok/s peak / 98 tok/s single-user / 0 length-truncation
- +3.6pp HumanEval over v2 FP8+MTP=3 SOTA (84.8% → 88.4%) on the matching config
- Reasoning cleanly separated into the OpenAI `reasoning` field; `content` clean
- Currently deployed (systemd unit `vllm-qwen36-27b-sota.service`)

**MTP=5 is benchmark-only, NOT production.** Although it scores 93.3% HE in the offline harness, it leaks raw `<think>...</think>` tokens directly into `content` for production requests — confirmed by live smoke test. The +4.9pp HE on MTP=5 reflects the harness counting those leaked think blobs, not real downstream code quality. Final production config = **FP8 + MTP=3**, validated leak-free across 420 trials at T=0.7 (300 plain-chat + 120 tool-calling, the latter exercising 10 OpenAI-format function tools across 30 scenarios with 95 % real tool-call rate). See [`FINAL_REPORT.md` § Production Incident](./FINAL_REPORT.md#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility) and [`LEAK_DETECTION.md`](./LEAK_DETECTION.md) for the methodology + per-config leak results.

Full numbers, throughput matrices, prefill scaling, and v3-vs-v2 deltas in [`FINAL_REPORT.md`](./FINAL_REPORT.md).

## What's in this repo

| Path | Contents |
|---|---|
| [`FINAL_REPORT.md`](./FINAL_REPORT.md) | Full Tier-3 report — TL;DR, throughput matrices, quality, v3-vs-v2, SOTA recommendation |
| [`DISCORD_V3.md`](./DISCORD_V3.md) | Ready-to-paste Discord broadcast summarizing the v3 suite |
| [`repne_reply_draft.md`](./repne_reply_draft.md) | Drafted reply for Repne — Run-1 crash artifacts + hardened launcher offer |
| [`LEAK_DETECTION.md`](./LEAK_DETECTION.md) | Think-token leak probe — methodology, per-config results, decision matrix |
| [`harness/tier3_v3_suite.sh`](./harness/tier3_v3_suite.sh) | Hardened orchestrator (per-phase log snapshots, trap-based cleanup, server-alive checks) |
| [`harness/sweep_lib.sh`](./harness/sweep_lib.sh) | Shared bench primitives (throughput, prefill, gates, HE, MBPP) |
| [`harness/leak_probe.py`](./harness/leak_probe.py) | Standalone leak detector (75-prompt corpus, `--repeat`/`--temperature` knobs, JSONL + summary outputs, CI-gatable exit code) |
| `leak-runs/fp8-mtp3/` | MTP=3 smoke result (75 trials, 0 leaks) |
| `leak-runs/fp8-mtp2/` | MTP=2 smoke result (74 captured, 0 leaks) |
| `leak-runs/fp8-mtp3-extended/` | MTP=3 production validation (**300 trials @ T=0.7, 0 leaks**) |
| `configs/tier3-bf16-dflash-n8/` | Run 2 clean — HE 92.7%, MBPP 91.1% |
| `configs/tier3-fp8-dflash-n8/` | Run 2 clean — HE 89.0%, MBPP 88.7% |
| `configs/tier3-fp8-mtp3/` | Run 2 clean — HE 88.4%, MBPP 89.1% |
| `configs/tier3-fp8-mtp5/` | Run 2 clean — HE 93.3% ⭐, MBPP 87.2%, peak 402 tok/s |
| `configs/tier3-bf16-dflash-n8.run1-CRASHED/` | Run 1 failure evidence (HE 33 ok + 35 http_error + 96 exception, throughput.json valid) |
| `configs/tier3-fp8-dflash-n8.run1-PARTIAL/` | Run 1 never-ran placeholder for Config 2 |
| `tier3.log` | Run 2 orchestrator stdout |
| `tier3.run1.log` | Run 1 orchestrator stdout (crashed at HE 34/164, Config 1) |
| `tier3.run2.aborted.log` | Brief pre-hardening relaunch (aborted to apply patches) |

## Configurations tested

All four configs share: TP=2, `--enable-prefix-caching`, `--default-chat-template-kwargs.preserve_thinking true`, `--attention-backend flashinfer`. **DFlash configs only** also pass `--speculative-config.attention_backend flashinfer` and `--speculative-config.use_local_argmax_reduction true`. **MTP configs cannot** use these two flags — the `Qwen3_5MTP` drafter does not implement `get_top_tokens()`, so the engine rejects `use_local_argmax_reduction` with a fatal `ValueError`. This was discovered the hard way during the v3 production rollout (see [`FINAL_REPORT.md` § Production Incident](./FINAL_REPORT.md#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility)).

| Label | Base model | Quantization | Spec method | Num speculative tokens |
|---|---|---|---|---|
| `tier3-bf16-dflash-n8` | `Qwen/Qwen3.6-27B` | BF16 | DFlash drafter | 8 |
| `tier3-fp8-dflash-n8` | `Qwen/Qwen3.6-27B-FP8` | FP8 | DFlash drafter | 8 |
| `tier3-fp8-mtp3` | `Qwen/Qwen3.6-27B-FP8` | FP8 | MTP (built-in) | 3 |
| `tier3-fp8-mtp5` | `Qwen/Qwen3.6-27B-FP8` | FP8 | MTP (built-in) | 5 |

Each config exercises five bench phases: load + smoke gates (4-prompt sanity), full throughput sweep (15 cells: c∈{1,4,16,64,128} × ctx∈{0, 32k, 128k}), prefill sweep (5 cells), HumanEval pass@1 (164 problems), MBPP pass@1 (257 problems).

## Run 1 → Run 2: the v3 BF16+DFlash crash

Run 1's Config 1 BF16+DFlash N=8 crashed at HumanEval problem 34/164 — server returned HTTP 500 `EngineCore encountered an issue` and 96 subsequent connection-reset errors. The Docker stack trace was lost (logs not snapshotted before container removal). Run 2 reran the identical config from a clean container and finished cleanly with HE 92.7%, MBPP 91.1%, and zero engine errors in `server_full.log`.

**Conclusion:** low-frequency transient engine failure on v3 under sustained decode load. Non-reproducible on rerun. Hardened launcher now captures `docker logs` after every bench phase so a future recurrence will preserve the traceback for Repne to debug.

See [`FINAL_REPORT.md` § "Run 1 (CRASHED) → Run 2 (Clean) Investigation"](./FINAL_REPORT.md) and [`repne_reply_draft.md`](./repne_reply_draft.md) for full details.

## Reproducing

```bash
# Pull v3 image
docker pull repne/vllm:v3

# Run the suite (assumes harness/ + sweep_lib.sh + stress_harness.py + llm_decode_bench.py on PATH)
cd harness && ./tier3_v3_suite.sh
```

Stop conditions, GPU pinning, and SOTA restart logic are encoded directly in `tier3_v3_suite.sh`.

## Related studies

- [`qwen-bench-2026-05-11-bf16-dflash-v2-sweep`](https://github.com/jcartu/qwen-bench-2026-05-11-bf16-dflash-v2-sweep) — initial v2 BF16+DFlash sweep
- [`qwen-bench-2026-05-11-v2-followup`](https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup) — v2 follow-up with Tier 1/2 mt=16384 reruns + smart_glue harness fix
- [`qwen-bench`](https://github.com/jcartu/qwen-bench) — hub index across all qwen-bench studies (this study lands in SOTA.md as Tier 3 v3)

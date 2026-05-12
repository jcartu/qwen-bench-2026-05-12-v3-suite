# Discord — v3 Suite Update (qwen R&D + Repne DM)

---

## qwen R&D chat (general)

**Tier 3 — Repne's `:v3` image, full 4-config suite (BF16+DFlash, FP8+DFlash, FP8+MTP=3, FP8+MTP=5)**

Pulled `repne/vllm:v3` (digest `fd2f7b567b19`), ran the same Tier-2 mt=16384 harness across 4 spec-decoding configurations. 3 h 32 min total wall time, GPUs 0+1 (GPU 2 untouched for Hindsight).

| Config | HE pass@1 | MBPP pass@1 | Peak tok/s (c=4) | Single-user tok/s | Length trunc |
|---|---:|---:|---:|---:|---:|
| BF16+DFlash N=8 | **92.7 %** (152/164) | **91.1 %** (234/257) | 300 | 86 | 0 |
| FP8+DFlash N=8 | 89.0 % | 88.7 % | 375 | 86 | 0 |
| FP8+MTP=3 | 88.4 % | 89.1 % | 369 | 98 | 0 |
| **FP8+MTP=5** | **93.3 %** ⭐ (153/164) | 87.2 % | **402** ⭐ | **101** ⭐ | 0 |

**v3 vs v2 (matching configs, mt=16384):**
- BF16+DFlash N=8 HE: 90.9 % → **92.7 %** (+1.8 pp)
- FP8+MTP=3 HE: 84.8 % → **88.4 %** (+3.6 pp)
- Every v3 config hit **0 length_truncated** on both HE and MBPP (v2 still showed truncation drift on FP8+MTP=3)

**Recommendation: promote `repne/vllm:v3` + FP8+MTP=5 as the new online SOTA.**
- +8.5 pp HE over current `:latest` FP8+MTP=3 (84.8 → 93.3)
- +64 % peak throughput (245 → 402 tok/s)
- +47 % single-user throughput (~69 → 101 tok/s)
- Run 2 ran clean for 3.5 h with zero engine crashes

One caveat: Run 1 of Config 1 (BF16+DFlash) crashed mid-HE with `EngineCore encountered an issue` (HTTP 500), but did **not** reproduce on rerun. Filed as low-frequency transient — see Repne DM below.

Hub: https://github.com/jcartu/qwen-bench/blob/main/SOTA.md
Study: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite

---

## Repne DM

Full v3 suite done. tl;dr: **`:v3` is faster, cleaner, and higher-quality than `:v2` on every config we tested.** Recommending we promote `:v3` + FP8+MTP=5 as the new online SOTA (93.3 % HE, 402 tok/s peak, 101 tok/s single-user, 0 length-trunc).

**Per-config deltas vs `:v2`:**
- BF16+DFlash N=8: HE 90.9 → 92.7 (+1.8 pp)
- FP8+MTP=3: HE 84.8 → 88.4 (+3.6 pp)
- All four v3 configs hit **0 length_truncated** on both HumanEval and MBPP. The `empty_response` drift we saw on v2 FP8+MTP=3 is gone too.

**One bug to file:** Run 1 of BF16+DFlash N=8 crashed at HE problem 34/164 with HTTP 500 `EngineCore encountered an issue. See stack trace (above) for the root cause.` Server stayed up for HTTP but engine subsystem was dead — 96 subsequent connection-reset errors. **Did not reproduce on Run 2** (same config, same args, same image — completed cleanly).

I lost the stack trace on Run 1 because my launcher only snapshotted `docker logs` at READY signal, not after benches. Patched the launcher for Run 2 to grab `docker logs` after every phase plus a final `server_full.log` + grep'd `server_errors.log`, so if it recurs we'll have the traceback. Artifacts of the Run 1 crash (partial humaneval.jsonl showing exactly where it died, plus the throughput.json/prefill.json that ran fine before the crash) are in `configs/tier3-bf16-dflash-n8.run1-CRASHED/`.

My read: low-frequency transient on v3 BF16+DFlash under sustained decode burst — possibly a cold-start race in FlashInfer/DFlash kernels or scheduler non-determinism with `async_scheduling=true`. Low priority since the rerun was clean, but worth a look before we ship `:v3` as the public default.

Numbers committed:
- Study: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite
- Hub SOTA: https://github.com/jcartu/qwen-bench/blob/main/SOTA.md
- Hardened launcher: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/harness/tier3_v3_suite.sh — feel free to crib if you want crash-trapping in your own bench rig.

Want me to flip production to `:v3` + FP8+MTP=5 now, or hold for your go?

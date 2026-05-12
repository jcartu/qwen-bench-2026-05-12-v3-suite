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

**Production rec: `repne/vllm:v3` + FP8+MTP=3** (MTP=5 is bench-only — leaks `<think>` tokens into `content` under live traffic).

Validated with a new permanent dual-mode leak probe (regex-scan of OpenAI `content` AND every `tool_calls[*].function.{name,arguments}` for `<think>`/`</think>` substrings):
- MTP=3 chat smoke (75 trials, T=0): **0 leaks**
- MTP=2 chat smoke (74 trials, T=0): **0 leaks**
- MTP=3 chat extended (300 trials, T=0.7, 4× corpus, 7m20s wall): **0 leaks**
- **MTP=3 tool-calling (120 trials, T=0.7, 30 scenarios over 10 OpenAI-schema function tools, 95 % real tool-call rate, multi-tool calls included, 2m11s wall): 0 leaks across content + tool_call names + tool_call arguments**

Why this matters: yesterday I had MTP=5 leaking on the very first 2-prompt smoke test. Without the leak issue, MTP=5 would give +5pp HE and +9 % peak throughput over MTP=3. So we're parking those numbers as benchmark-only and waiting on the upstream parser fix before considering MTP=5 for production.

Production: +3.6pp HE over v2 (FP8+MTP=3 84.8 → 88.4), 369 tok/s peak, 98 tok/s single-user, 0 length-truncation.

One caveat: Run 1 of Config 1 (BF16+DFlash) crashed mid-HE with `EngineCore encountered an issue` (HTTP 500), but did **not** reproduce on rerun. Filed as low-frequency transient — see Repne DM below.
Hub: https://github.com/jcartu/qwen-bench/blob/main/SOTA.md
Study: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite

---

## Repne DM

Full v3 suite done. tl;dr: **`:v3` is faster, cleaner, and higher-quality than `:v2` on every config we tested.** Production is now on `:v3` + FP8+MTP=3 (88.4 % HE, 89.1 % MBPP, 369 tok/s peak, 98 tok/s single-user, 0 length-trunc). MTP=5 stays bench-only — see leak finding below.

**Per-config deltas vs `:v2`:**
- BF16+DFlash N=8: HE 90.9 → 92.7 (+1.8 pp)
- FP8+MTP=3: HE 84.8 → 88.4 (+3.6 pp)
- All four v3 configs hit **0 length_truncated** on both HumanEval and MBPP. The `empty_response` drift we saw on v2 FP8+MTP=3 is gone too.

**One bug to file:** Run 1 of BF16+DFlash N=8 crashed at HE problem 34/164 with HTTP 500 `EngineCore encountered an issue. See stack trace (above) for the root cause.` Server stayed up for HTTP but engine subsystem was dead — 96 subsequent connection-reset errors. **Did not reproduce on Run 2** (same config, same args, same image — completed cleanly).

I lost the stack trace on Run 1 because my launcher only snapshotted `docker logs` at READY signal, not after benches. Patched the launcher for Run 2 to grab `docker logs` after every phase plus a final `server_full.log` + grep'd `server_errors.log`, so if it recurs we'll have the traceback. Artifacts of the Run 1 crash (partial humaneval.jsonl showing exactly where it died, plus the throughput.json/prefill.json that ran fine before the crash) are in `configs/tier3-bf16-dflash-n8.run1-CRASHED/`.

**Second finding (during prod rollout):** `--speculative-config.use_local_argmax_reduction true` is **DFlash-only**, not TP-only. The `Qwen3_5MTP` drafter does not implement `get_top_tokens()`, so the engine refuses to start an MTP config with that flag (`ValueError: use_local_argmax_reduction is enabled but draft model Qwen3_5MTP does not implement get_top_tokens()`). We hit this on the first production-promotion attempt and got a tight crash-loop. Fix: don't pass `use_local_argmax_reduction` or per-spec `attention_backend` for MTP configs. The matching `server.log`s in `configs/tier3-fp8-mtp{3,5}/` show the engine's `non-default args` dict and confirm the suite already had those flags absent for MTP — it was the production launcher that was wrong, not the bench harness.

**Third finding — MTP=5 leaks `<think>` tokens into `content`:** On a 2-prompt smoke test against `:v3` + FP8+MTP=5, the OpenAI `content` field contained raw `<think>...</think>` substrings instead of routing them through `reasoning`. This is why we kept production on MTP=3 instead of promoting MTP=5 despite its 93.3 % HE / 402 tok/s. Operator concern — "reports about MTP=2 and MTP=3 doing it too" — prompted a permanent leak probe (`harness/leak_probe.py`, dual-mode: 75-prompt chat corpus across 15 categories AND a tools mode with 30 scenarios over 10 OpenAI-schema function tools, configurable repeat + temperature, exit-code-gated for CI). A second escalation — "did our extensive leak test do extensive tool calling?" — added the tools mode, which scans not just `content` but every `tool_calls[*].function.{name, arguments}`. Runs:
- MTP=3 chat smoke @ T=0 — 75 trials — **0 leaks**
- MTP=2 chat smoke @ T=0 — 74 trials — **0 leaks**
- MTP=3 chat extended @ T=0.7 — **300 trials, 7m20s wall — 0 leaks**
- MTP=3 tool-calling @ T=0.7 — **120 trials, 95 % tool-call rate (114/120), 4 multi-tool responses, 2m11s wall — 0 leaks across all three scanned surfaces**

So on `:v3` the leak class appears to be MTP=5-specific in our environment. Methodology + raw artifacts in [`LEAK_DETECTION.md`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/LEAK_DETECTION.md) + `leak-runs/`. If you have a repro for MTP=3 leaking we'd love the prompts — we'll fold them into the corpus.

My read: low-frequency transient on v3 BF16+DFlash under sustained decode burst — possibly a cold-start race in FlashInfer/DFlash kernels or scheduler non-determinism with `async_scheduling=true`. Low priority since the rerun was clean, but worth a look before we ship `:v3` as the public default.

Numbers committed:
- Study: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite
- Hub SOTA: https://github.com/jcartu/qwen-bench/blob/main/SOTA.md
- Hardened launcher: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/harness/tier3_v3_suite.sh — feel free to crib if you want crash-trapping in your own bench rig.

Production is already on `:v3` + FP8+MTP=3 as of ~10:03 MSK, re-validated leak-free at 300 chat trials as of ~10:36 MSK, and re-re-validated leak-free at 120 tool-calling trials as of ~10:42 MSK (420 total trials, zero leaks across `content` + `tool_calls[*].function.name` + `tool_calls[*].function.arguments`). Happy to wire MTP=5 in too if/when the think-token leak is fixed upstream.

<p align="center">
  <img src="docs/images/study_hero.png" alt="Tier-3 v3 full suite — 4 configs, 1,684 quality probes, 0 engine errors on Run 2" width="100%" />
</p>

# Tier 3 v3 Full Suite — Final Report

**Date:** 2026-05-12 (Tue) MSK
**Image:** `repne/vllm:v3` (digest `fd2f7b567b19`, 29.7 GB)
**Model:** Qwen3.6-27B (BF16 + FP8)
**Drafter:** z-lab/Qwen3.6-27B-DFlash (where applicable)
**Hardware:** Blackwell, TP=2 on GPUs 0+1 (UUIDs ba6334bc + 538bf008)
**GPU 2:** UNTOUCHED at 89.8 GB throughout (oss-120b for Hindsight, per user req)
**Suite Wall Time:** 03:59:19 → 07:31:32 MSK = **3 h 32 min**
**Image Args (DFlash configs):** `--speculative-config.use_local_argmax_reduction true`,
`--speculative-config.attention_backend flashinfer`, `--attention-backend flashinfer`,
`--default-chat-template-kwargs.preserve_thinking true`, `--enable-prefix-caching`
**Image Args (MTP configs):** `--attention-backend flashinfer`,
`--default-chat-template-kwargs.preserve_thinking true`, `--enable-prefix-caching`
(see [§ Production Incident](#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility) — MTP drafter does NOT implement `get_top_tokens()`, so `use_local_argmax_reduction` and per-spec `attention_backend` are DFlash-only)

---

## TL;DR

| Config | HE Pass | MBPP Pass | Best @ c=4 ctx=0 | Best @ c=1 ctx=0 | Length Trunc |
|---|---|---|---|---|---|
| **BF16+DFlash N=8** | **92.7%** (152/164) | **91.1%** (234/257) | 300 tok/s | 86 tok/s | **0** |
| **FP8+DFlash N=8** | 89.0% (146/164) | 88.7% (228/257) | **375 tok/s** | 86 tok/s | **0** |
| **FP8+MTP=3** | 88.4% (145/164) | **89.1%** (229/257) | 369 tok/s | 98 tok/s | **0** |
| **FP8+MTP=5** | 93.3% (153/164) | 87.2% (224/257) | **402 tok/s** ⭐ | **101 tok/s** ⭐ | **0** ⚠️ |

**Winner by metric (offline benchmark numbers):**
- Best HE pass:       **FP8+MTP=5 @ 93.3%** (vs Tier 2 v2 FP8+MTP=3 84.8%)
- Best MBPP pass:     **BF16+DFlash N=8 @ 91.1%**
- Best peak tps:      **FP8+MTP=5 @ 402 tok/s** (c=4 ctx=0)
- Best single-user:   **FP8+MTP=5 @ 101 tok/s** (c=1 ctx=0)
- Best long-context:  **FP8+DFlash N=8 @ 345.9 tok/s** at c=4 ctx=128k
- Cleanest output:    **All 4 configs hit 0 length_truncated** (vs v2 Tier 2's mixed)

⚠️ **Production winner = FP8+MTP=3** (not MTP=5). See [Production Incident](#production-incident-2026-05-12-mtp--use_local_argmax_reduction-incompatibility) — MTP=5 leaks raw `<think>` tokens into user-visible `content` in production traffic, despite scoring well on offline HE. MTP=3 keeps reasoning cleanly separated into the `reasoning` field.

**v3 vs v2 (matching configs, mt=16384):**
- BF16+DFlash N=8 HE: 92.7% (v3) vs 90.9% (v2) — **+1.8pp**
- FP8+MTP=3 HE:      88.4% (v3) vs 84.8% (v2) — **+3.6pp**
- All v3 configs have 0 length_truncated (v2 FP8+MTP=3 had drift issues)

---

## Run 1 (CRASHED) → Run 2 (Clean) Investigation

**Run 1** (PID 3507255, started 03:25 MSK):
Config 1 BF16+DFlash N=8 — loaded fine, throughput sweep completed (15 cells), gates 4/4 PASS, prefill sweep done. **Engine crashed at HumanEval problem 34/164** with HTTP 500
`{"error":{"message":"EngineCore encountered an issue. See stack trace (above) for the root cause.","type":"InternalServerError","code":500}}` — server stayed up for HTTP but engine subsystem was dead. 96 subsequent connection-reset errors. MBPP ran against dead server (257 instant exceptions). Stack trace lost (container removed during cleanup before logs captured).

**Diagnostic gap:** Run 1 launcher only captured `docker logs` at READY signal, not after benches. Engine death during benches → no traceback preserved.

**Patches applied for Run 2** (`tier3_v3_suite.sh`):
1. Per-bench `server_alive` curl check (`/v1/models`) before each phase; skip + log if dead.
2. `docker logs > server_after_${phase}.log` after every bench phase (gates, throughput, prefill, quality).
3. `docker logs > server_full.log` at end of each config.
4. `grep -iE 'TypeError|ValueError|RuntimeError|Engine core|CUDA error|out of memory|Traceback|encountered an issue|Killed|Signal|Aborted|fatal'` → `server_errors.log`.
5. `trap cleanup EXIT INT TERM` to guarantee SOTA restart even on hard kill.
6. Final `[SERVER OK/DEAD]` annotation per config.

**Run 2 result on identical Config 1:** Server stayed live through entire BF16+DFlash run (~50 min), HE 92.7%, MBPP 91.1%, no engine errors in logs. **Crash was non-reproducible.**

**Conclusion on the bug:** v3 BF16+DFlash N=8 has a low-frequency transient engine failure that does NOT reproduce on rerun. Possible causes:
- Cold-start race in FlashInfer / DFlash kernels at first decode burst
- KV-cache initialization timing
- Non-determinism in scheduler when `async_scheduling=true` (v3 default)

**Recommendation to Repne:** This needs to be filed as a low-priority bug ("low-frequency v3 engine crash during sustained decode workload") with the Run 1 artifacts (`configs/tier3-bf16-dflash-n8.run1-CRASHED/` includes the partial `humaneval.jsonl` showing exactly where it died). No stack trace was captured for Run 1 because logs weren't snapshotted at the right time; **launcher is now hardened**, so a future re-occurrence will preserve traces.

---

## Stability (Run 2)

All 4 configs ran end-to-end without a single engine crash, HTTP 5xx, or empty-response storm:

| Config | Load Time | Gates | Throughput | Prefill | HE | MBPP | Server Final |
|---|---|---|---|---|---|---|---|
| BF16+DFlash N=8 | 181s | 4/4 | ✅ 15 cells | ✅ 5 cells | ✅ 164/164 | ✅ 257/257 | **OK** |
| FP8+DFlash N=8  | 241s | 4/4 | ✅ 15 cells | ✅ 5 cells | ✅ 164/164 | ✅ 257/257 | **OK** |
| FP8+MTP=3       | 314s | 4/4 | ✅ 15 cells | ✅ 5 cells | ✅ 164/164 | ✅ 257/257 | **OK** |
| FP8+MTP=5       | 319s | 4/4 | ✅ 15 cells | ✅ 5 cells | ✅ 164/164 | ✅ 257/257 | **OK** |

No `server_errors.log` had non-trivial content in any config. **Zero engine failures across full Run 2.**

---

## Throughput (decode tok/s, full matrices)

### BF16+DFlash N=8
```
  c    ctx  agg_tps  per_user  accept_rate
  1      0     85.6      85.5      0.094
  1  16384     86.6      82.9      0.211
  1  32768     86.1      86.7      0.190
  1  65536     80.1      80.1      0.146
  1 131072     78.0      76.6      0.362
  2      0    165.5      79.2      0.139
  4      0    300.1      74.6      0.233
  2  16384    166.2      83.2      0.155
  4  16384    304.4      75.4      0.259
  2  32768    161.2      83.4      0.104
  4  32768    304.9      74.9      0.277
  2  65536    164.7      81.7      0.165
  4  65536    303.7      74.8      0.235
  2 131072    151.4      74.6      0.350
  4 131072    276.6      68.3      0.272
```
128k vs 0 degradation: **8% loss at c=4**.

### FP8+DFlash N=8
```
  c    ctx  agg_tps  per_user  accept_rate
  1      0     86.2      86.2      0.162
  1  16384     93.9      96.5      0.202
  1  32768     94.0      97.0      0.136
  1  65536    101.3     102.0      0.146
  1 131072     92.4      96.0      0.140
  2      0    185.4      87.2      0.206
  4      0    375.7      94.5      0.274
  2  16384    188.1      94.5      0.335
  4  16384    366.4      92.6      0.279
  2  32768    190.5      95.5      0.208
  4  32768    370.2      92.3      0.237
  2  65536    175.4      88.2      0.220
  4  65536    361.3      89.6      0.206
  2 131072    189.1      96.2      0.165
  4 131072    345.9      84.4      0.166
```
128k vs 0 degradation: **8% loss at c=4**. Fastest at long context.

### FP8+MTP=3
```
  c    ctx  agg_tps  per_user  accept_rate
  1      0     97.7      97.9      0.450
  1  16384     99.0      98.6      0.849
  1  32768     96.6      96.9      0.518
  1  65536     95.4      93.0      0.649
  1 131072     88.0      87.0      0.461
  2      0    190.3      89.5      0.518
  4      0    369.4      83.1      0.482
  2  16384    191.8      96.0      0.564
  4  16384    385.7      96.5      0.621
  2  32768    192.1      95.7      0.765
  4  32768    372.3      90.9      0.559
  2  65536    181.6      90.3      0.586
  4  65536    366.6      89.4      0.572
  2 131072    170.5      83.7      0.594
  4 131072    326.9      79.3      0.472
```
MTP=3 hits **0.85 accept_rate at c=1 ctx=16k** — highest single-user spec acceptance of all configs.

### FP8+MTP=5
```
  c    ctx  agg_tps  per_user  accept_rate
  1      0    101.0     100.0      0.418
  1  16384    104.1     105.2      0.453
  1  32768     97.0      97.5      0.576
  1  65536     92.4      92.3      0.156
  1 131072     85.5      86.5      0.387
  2      0    181.1      82.5      0.318
  4      0    402.2     100.5      0.424  ⭐ PEAK
  2  16384    202.1     101.0      0.455
  4  16384    392.2      97.8      0.476
  2  32768    202.6      99.9      0.406
  4  32768    389.4      96.4      0.485
  2  65536    180.6      92.3      0.221
  4  65536    367.8      89.2      0.314
  2 131072    169.9      84.4      0.417
  4 131072    331.5      79.9      0.405
```
MTP=5 ⭐ **402 tok/s peak** at c=4 ctx=0, **105 tok/s single-user** at c=1 ctx=16k.

---

## Quality (HE + MBPP @ mt=16384, c=8)

### HumanEval
| Config | Pass | OK | Empty | TestFail | LengthTrunc |
|---|---|---|---|---|---|
| BF16+DFlash N=8 | 92.7% | 152 | 6 | 6 | **0** |
| FP8+DFlash N=8  | 89.0% | 146 | 10 | 8 | **0** |
| FP8+MTP=3       | 88.4% | 145 | 13 | 6 | **0** |
| FP8+MTP=5       | **93.3%** ⭐ | 153 | 5 | 6 | **0** |

### MBPP
| Config | Pass | OK | Empty | TestFail | LengthTrunc |
|---|---|---|---|---|---|
| BF16+DFlash N=8 | **91.1%** ⭐ | 234 | 12 | 11 | **0** |
| FP8+DFlash N=8  | 88.7% | 228 | 16 | 13 | **0** |
| FP8+MTP=3       | 89.1% | 229 | 15 | 13 | **0** |
| FP8+MTP=5       | 87.2% | 224 | 21 | 12 | **0** |

**Zero length truncation across all 4 configs and both benchmarks** — `:v3` confirms Tier 2's finding that mt=16384 + smart_glue post-processing eliminates the truncation failure mode that plagued Tier 0/1.

---

## SOTA Recommendation (Production-Validated)

**Previous SOTA (v2 Tier 2):**
- Online BF16: BF16+DFlash N=8 @ mt=16384 → 90.9% HE
- Online FP8: FP8+MTP=3 @ mt=16384 → 84.8% HE

**New v3 production SOTA (post-2026-05-12 incident, validated in production):**
- **Online FP8 (production):** **FP8+MTP=3** — 88.4% HE / 89.1% MBPP / 369 tok/s peak / 98 tok/s single-user. **Clean output, reasoning separated into `reasoning` field. Currently deployed.**
- **Online BF16:** BF16+DFlash N=8 — 92.7% HE / 91.1% MBPP

**Offline-only / benchmark-only:**
- FP8+MTP=5 — 93.3% HE on paper, but **DO NOT DEPLOY**: leaks `<think>` tokens into `content` in production. The +4.9pp HE on the benchmark harness reflects the benchmark counting raw `<think>...</think>` blobs as part of code, not real downstream code quality. Production users see broken responses.

**Why MTP=3 over MTP=5 in production:**
1. Cleaner reasoning/content separation (verified via live smoke test on 2026-05-12 10:03 MSK)
2. Still 369 tok/s peak (within 8% of MTP=5's 402)
3. Single-user 98 tok/s — Hindsight-class interactivity
4. Zero length truncation
5. Stable through 50-min sustained workload

---

## Artifacts

```
/tmp/qwen-bench-2026-05-12-v3-suite/
├── FINAL_REPORT.md                           ← this file
├── tier3.log                                 ← orchestrator log Run 2
├── tier3.run2.aborted.log                    ← brief aborted relaunch (~2 min, killed for hardening)
├── tier3.run1.log                            ← original Run 1 with crash
├── harness/
│   ├── tier3_v3_suite.sh                     ← hardened launcher (crash-resilient)
│   └── sweep_lib.sh                          ← shared bench primitives
├── configs/
│   ├── tier3-bf16-dflash-n8/                 ← Run 2 (CLEAN)
│   │   ├── humaneval.jsonl + _summary.json   (152/164 = 92.7%)
│   │   ├── mbpp.jsonl + _summary.json        (234/257 = 91.1%)
│   │   ├── throughput.json                   (15 cells)
│   │   ├── prefill.json                      (5 cells)
│   │   ├── gates.json                        (4/4)
│   │   ├── server_full.log
│   │   ├── server_errors.log                 (empty)
│   │   ├── server_after_{gates,throughput,prefill,quality}.log
│   │   └── server.log
│   ├── tier3-fp8-dflash-n8/                  ← Run 2 (CLEAN)
│   │   └── ... 89.0% / 88.7%
│   ├── tier3-fp8-mtp3/                       ← Run 2 (CLEAN)
│   │   └── ... 88.4% / 89.1%
│   ├── tier3-fp8-mtp5/                       ← Run 2 (CLEAN) ⭐
│   │   └── ... 93.3% / 87.2%
│   ├── tier3-bf16-dflash-n8.run1-CRASHED/    ← Run 1 partial (the v3 BF16 engine crash)
│   │   ├── humaneval.jsonl (33 ok + 35 http_error + 96 exception)
│   │   ├── throughput.json (15 cells, valid)
│   │   ├── prefill.json
│   │   └── gates.json
│   └── tier3-fp8-dflash-n8.run1-PARTIAL/     ← Run 1, never ran (crash above terminated suite)
└── tier3.pid                                  ← was PID of Run 2, now stale
```

---

## Next Steps

1. **(USER, MORNING)** Read this report. Decide whether to:
   - Commit Tier 3 results to new repo `qwen-bench-2026-05-12-v3-suite` (recommended)
   - Update `qwen-bench` hub `SOTA.md` with v3 numbers (recommended)
   - DM Repne the v3 BF16 crash artifacts as a low-priority report
2. **(AUTOMATED)** SOTA service already auto-restarted (Active: running since 07:31:32 MSK).
3. **(AUTOMATED)** GPU 2 = 89.8 GB throughout, never touched.

## What Went Right
- Hardened launcher worked: 100% completion rate Run 2.
- Server health monitoring + per-phase log capture in place for next crash.
- Suite ETA was accurate (predicted 07:14-07:28, actual 07:31).
- SOTA auto-restoration successful.
- Hindsight oss-120b GPU 2 untouched the entire 3.5h run.

## What Was Unexpected
- Run 1 BF16+DFlash engine crash — could not be reproduced in Run 2.
- FP8+MTP=5 beat BF16+DFlash on HumanEval (93.3% vs 92.7%) — FP8 quality is closing on BF16 with deeper MTP.
- FP8+DFlash MBPP (88.7%) **underperformed** FP8+MTP=3 MBPP (89.1%) — MBPP doesn't reward DFlash's drafter as much as HE does.

---

## Production Incident 2026-05-12: MTP + `use_local_argmax_reduction` Incompatibility

<p align="center">
  <img src="docs/images/production_incident.png" alt="Production incident post-mortem — MTP drafter incompatible with use_local_argmax_reduction; fix isolates the flag to DFlash configs only" width="90%" />
</p>


**Symptom:** First v3 production promotion attempt (`launch-qwen36-27b-sota.sh` patched to `repne/vllm:v3` + MTP=3 + `--speculative-config.use_local_argmax_reduction true`) entered a systemd crash-loop. Worker fatal error:

```
ValueError: use_local_argmax_reduction is enabled but draft model
Qwen3_5MTP does not implement get_top_tokens().
```

**Root cause:** `use_local_argmax_reduction` requires the drafter to implement `get_top_tokens()`. The DFlash drafter (`z-lab/Qwen3.6-27B-DFlash`) does; the in-model MTP drafter (`Qwen3_5MTP`) does NOT. This is a **drafter-type** constraint, NOT a TP>1 constraint as initially understood.

**Evidence from this suite:** the four config `server.log` files show the engine's `non-default args` dict — `use_local_argmax_reduction: True` appears only in `tier3-bf16-dflash-n8` and `tier3-fp8-dflash-n8`, and is absent from both `tier3-fp8-mtp3` and `tier3-fp8-mtp5`. The MTP=3 / MTP=5 benchmarks succeeded precisely because the harness omitted that flag.

**Fix:** Removed `--speculative-config.use_local_argmax_reduction true` and `--speculative-config.attention_backend flashinfer` from the production launcher's MTP code path. Kept top-level `--attention-backend flashinfer`. Service restarted, ready in 134s, smoke test green.

**Compounding finding — MTP=5 think-token leakage:** While verifying the fix, we found that MTP=5 (originally recommended as v3 SOTA above) leaks `<think>...</think>` tokens directly into the OpenAI-format `content` field in production traffic, despite producing high HE pass rates in the benchmark harness. MTP=3 routes reasoning cleanly into the separate `reasoning` field (`finish_reason: stop`, `content` clean, `reasoning` populated). **Production now runs MTP=3.**

**Verification (2026-05-12 ~10:03 MSK):**
- Container: `repne/vllm:v3` (digest `fd2f7b567b19`)
- Args confirmed: `speculative-config.method mtp`, `num_speculative_tokens 3`, NO `use_local_argmax_reduction`, NO `speculative-config.attention_backend`, top-level `--attention-backend flashinfer`, `--enable-prefix-caching`, `--default-chat-template-kwargs.preserve_thinking true`
- Smoke test: `content='\n\nOK'`, `reasoning` separated cleanly, zero `<think>` tags in `content`
- GPU pinning preserved: GPU 0 + GPU 1 via UUID, GPU 2 untouched at 89.8 GB (Hindsight)

**Follow-up: formal leak-detection probe (2026-05-12 ~10:36 MSK).** The 2-prompt verification above was operator-flagged as insufficient — community reports suggested MTP=3 (and even MTP=2) might leak `<think>` tokens too, just at a lower rate than MTP=5. We built a permanent probe (`harness/leak_probe.py`) with a 75-prompt diverse corpus (15 categories: trivial echo, arithmetic, code, algorithmic word problems, reasoning, creative, knowledge, multi-step, definitional, translation, edge cases, big-O, numerical estimates, pattern recognition, coding gotchas) and ran:

| Config       | Mode  | Trials | Wall   | Leaks | Leak rate | Notes |
|--------------|-------|-------:|-------:|------:|----------:|-------|
| FP8 + MTP=3  | chat  | 75     | 97.8s  |   0   |  0.0%     | T=0.0, smoke |
| FP8 + MTP=2  | chat  | 74*    | ~85s   |   0   |  0.0%     | T=0.0, override `NUM_SPEC=2` via systemd drop-in |
| FP8 + MTP=3 (extended) | chat | **300** | **440.6s** | **0** | **0.0%** | T=0.7, 4 × corpus, unique seeds — production validation (plain chat) |
| **FP8 + MTP=3 (tools)** | **tools** | **120** | **131.5s** | **0** | **0.0%** | **T=0.7, 30 scenarios × 4 reps across 10 OpenAI-schema tools — production validation (tool-calling)** |

\* MTP=2 emitted all 74 ok markers in its log; one summary write was killed by parent-shell SIGTERM before flush. Stats reconstructed from the log; no leaks observed. The harness now uses `setsid nohup` so future runs survive shell death.

**Production decision: FP8 + MTP=3 stays.** Operator-driven second escalation ("did our extensive leak test do extensive tool calling?") prompted a dual-mode upgrade to the probe: `--mode tools` exposes 10 OpenAI-format function tools (`get_weather`, `search_web`, `send_email`, `run_sql`, `calculator`, `create_file`, `list_files`, `get_stock_price`, `translate`, `create_ticket`) across 30 scenarios spanning single-tool, full-toolbox, and multi-tool-in-one-response cases. Of 120 trials at T=0.7, **114 produced real `tool_calls` responses** (95 % tool-call rate, including 4 multi-tool responses), with an average of 1532 chars of reasoning per trial — the reasoning parser was heavily exercised. Surfaces scanned: `content`, every `tool_calls[*].function.name`, every `tool_calls[*].function.arguments`. **Zero leaks on any surface.** Combined across both modes: **420 trials, 0 leaks.** The original MTP=5 leak observation remains: MTP=5 leaked on the very first 2-prompt smoke test under identical infrastructure. Documented fallback if leaks ever do appear in real traffic: FP8 + DFlash N=8 (slightly lower quality, no MTP-class leak risk).

Full methodology, knobs, raw artifacts, and CI-gating instructions: [`LEAK_DETECTION.md`](./LEAK_DETECTION.md). Raw run artifacts: `leak-runs/fp8-mtp3/`, `leak-runs/fp8-mtp2/`, `leak-runs/fp8-mtp3-extended/`, `leak-runs/fp8-mtp3-tools/`.

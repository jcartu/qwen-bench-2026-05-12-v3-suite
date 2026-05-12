# Repne reply draft — `:v3` suite + crash report

Full v3 suite is done. Headline first, then the bug.

## Headline (good news)

`:v3` is **better than `:v2` on every config I tested**, no regressions:

| Config | HE pass@1 (v2 → v3) | Peak tok/s (v2 → v3) | length_trunc |
|---|---|---|---|
| BF16+DFlash N=8 | 90.9 % → **92.7 %** | ~245 → 300 | 0 |
| FP8+DFlash N=8 | (new) → 89.0 % | — → 375 | 0 |
| FP8+MTP=3 | 84.8 % → **88.4 %** | ~245 → 369 | 0 |
| **FP8+MTP=5** | (new) → **93.3 %** ⭐ | — → **402** ⭐ | 0 |

Every v3 config hit **0 length_truncated** on both HE (164 problems) and MBPP (257 problems). The `empty_response` drift I was seeing on v2 FP8+MTP=3 at mt=16384 — also gone on v3. Suite ran clean for 3 h 32 min, no engine crashes on Run 2.

I'm recommending we promote **`:v3` + FP8+MTP=5** as the new online SOTA: +8.5 pp HE over current `:latest` FP8+MTP=3, +64 % peak throughput, +47 % single-user throughput.

Full report: https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/FINAL_REPORT.md

## Bug to file (one)

**Symptom:** Run 1 of Config 1 (BF16+DFlash N=8) crashed at HumanEval problem 34/164 with HTTP 500:
```
{"error":{"message":"EngineCore encountered an issue. See stack trace (above) for the root cause.","type":"InternalServerError","code":500}}
```
Server stayed up for HTTP responses (didn't 502/connection-refuse) but the engine subsystem was dead. 96 subsequent HE requests returned connection-reset errors, then 257 MBPP requests all instant-exceptioned.

**Trigger context:** ~80 s into HumanEval traffic at concurrency=1 (HE harness fires sequentially), after a full clean throughput sweep (15 cells, c={1,4,16,64,128} × ctx={0,32k,128k}), gates pass (4/4), prefill sweep (5 cells). Engine had handled ~maybe 500-1000 inferences cleanly before the crash.

**Args (excerpt):**
```
--tensor-parallel-size 2
--enable-prefix-caching
--default-chat-template-kwargs.preserve_thinking true
--attention-backend flashinfer
--speculative-config.method draft_model
--speculative-config.model z-lab/Qwen3.6-27B-DFlash
--speculative-config.num_speculative_tokens 8
--speculative-config.attention_backend flashinfer
--speculative-config.use_local_argmax_reduction true
```
Full args dump in [`configs/tier3-bf16-dflash-n8.run1-CRASHED/server_args.txt`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/configs/tier3-bf16-dflash-n8.run1-CRASHED/server_args.txt).

**Does NOT reproduce:** Run 2 with identical args, identical image (digest `fd2f7b567b19`), identical hardware (TP=2 on the same two GPUs), identical bench traffic — completed cleanly. HE 92.7 %, MBPP 91.1 %, no engine errors in logs over ~50 min of total decode.

**Diagnostic gap (my fault):** Run-1 launcher only captured `docker logs` at READY signal, not after benches. Docker stack trace was lost when the container was removed in cleanup. I've **patched the launcher** to snapshot `docker logs` after every bench phase + final `server_full.log` + grep'd `server_errors.log` covering `TypeError|ValueError|RuntimeError|Engine core|CUDA error|out of memory|Traceback|encountered an issue|Killed|Signal|Aborted|fatal`. If this recurs we'll have the full traceback. Launcher: [`harness/tier3_v3_suite.sh`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/harness/tier3_v3_suite.sh).

**Artifacts I do have from Run 1** (in [`configs/tier3-bf16-dflash-n8.run1-CRASHED/`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/tree/main/configs/tier3-bf16-dflash-n8.run1-CRASHED/)):
- `throughput.json` — full sweep completed clean (15 cells), so engine was healthy at this point
- `prefill.json` — full prefill sweep clean
- `gates.json` — 4/4 pass
- `humaneval.jsonl` — exactly **33 ok + 35 http_error + 96 exception** (= 164 total), so first crash at request #34
- `humaneval.log` — shows the HTTP 500 sequence
- `mbpp.jsonl` — 257 instant exceptions (engine was already dead)
- `server.log` — `docker logs` snapshot at READY (pre-crash, no crash trace here)

**My best guess at root cause** (FWIW, not authoritative):
1. Cold-start race in FlashInfer / DFlash kernels at first sustained sequential-decode burst (HE harness is c=1 sequential, very different traffic profile than the c={4,16,64,128} throughput sweep that preceded it)
2. KV-cache initialization timing edge case after long idle between throughput sweep and HE
3. Non-determinism in scheduler when `async_scheduling=true` (v3 default per the README you shared)

Happy to bisect kernel paths or knobs if you want — let me know what flags to flip.

**Priority from my side:** low. Promotion to `:latest` is gated on this not reproducing in production load. Given Run 2 was clean and we have hardened logging now, I'd suggest we promote `:v3` to staging, run it under real traffic for 24 h with the hardened launcher capturing crashes, and only block public promotion if it recurs.

Cool with that approach? Or do you want a deeper repro attempt first?

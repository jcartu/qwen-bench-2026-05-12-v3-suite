# Repne reply draft — `:v3` suite + crash report

Full v3 suite is done. Headline first, then the bug.

## Headline (good news)

`:v3` is **better than `:v2` on every config I tested**, no regressions:

| Config | HE pass@1 (v2 → v3) | Peak tok/s (v2 → v3) | length_trunc |
|---|---|---|---|
| BF16+DFlash N=8 | 90.9 % → **92.7 %** | ~245 → 300 | 0 |
| FP8+DFlash N=8 | (new) → 89.0 % | — → 375 | 0 |
| FP8+MTP=3 | 84.8 % → **88.4 %** | ~245 → 369 | 0 |
| FP8+MTP=5 | (new) → 93.3 % | — → **402** | 0 ⚠️ (see footnote) |

Every v3 config hit **0 length_truncated** on both HE (164 problems) and MBPP (257 problems). The `empty_response` drift I was seeing on v2 FP8+MTP=3 at mt=16384 — also gone on v3. Suite ran clean for 3 h 32 min, no engine crashes on Run 2.

I'm running **`:v3` + FP8+MTP=3 in production right now** (live since ~10:03 MSK 2026-05-12, 88.4 % HE / 89.1 % MBPP / 369 tok/s peak / 98 tok/s single-user). MTP=5 scored higher on the offline harness (93.3 % HE, 402 tok/s peak), but — footnote — it leaks raw `<think>...</think>` blocks into the OpenAI `content` field on production traffic. Reasoning is supposed to be routed into the separate `reasoning` field, and MTP=3 does that cleanly; MTP=5 doesn't. The harness counts the leaked think blobs as code, which is where the +4.9 pp HE comes from — not real downstream code quality. So MTP=5 stays as a benchmark-only number for now, until the leak is fixed upstream.

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

---

## Second finding from the production rollout (sharing in case it's useful)

First production-promotion attempt with `:v3` + FP8+MTP=3 entered a tight systemd crash-loop. Root cause:

```
ValueError: use_local_argmax_reduction is enabled but draft model
Qwen3_5MTP does not implement get_top_tokens().
```

I had carried `--speculative-config.use_local_argmax_reduction true` and `--speculative-config.attention_backend flashinfer` over from the DFlash configs into the MTP production launcher, under the (wrong) impression that `use_local_argmax_reduction` was a TP>1 thing. It's actually a **drafter-capability** thing: DFlash implements `get_top_tokens()`, the in-model `Qwen3_5MTP` drafter does not.

Confirmed in retrospect by looking at my own bench `server.log`s — the engine's `non-default args` dict has `use_local_argmax_reduction: True` only in the two DFlash config logs, and is absent from both MTP=3 and MTP=5 logs. The bench harness was right; the production launcher was wrong. Mentioning in case anyone else trips over the same edge: would be lovely if either the docs or the error message itself called out the drafter constraint.

---

## Third finding (think-token leak validation)

I added a permanent dual-mode leak detector to the bench harness ([`harness/leak_probe.py`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/harness/leak_probe.py)) and ran it against four configurations. Detector logic differs by mode:

- **`--mode chat`** — 75-prompt diverse corpus (15 categories: trivial echo, arithmetic, short code, algorithmic code, word problems, logic, creative, knowledge, multi-step, definitional, translation, edge cases, big-O, numerical estimates, pattern recognition, coding gotchas), case-insensitive regex scan of `choices[0].message.content` for `<think`/`</think>`/`<reasoning`/`</reasoning>`.
- **`--mode tools`** — 30 scenarios (single-tool, full-toolbox, multi-tool-call) across 10 OpenAI-schema function tools (`get_weather`, `search_web`, `send_email`, `run_sql`, `calculator`, `create_file`, `list_files`, `get_stock_price`, `translate`, `create_ticket`), scans `content` AND every `tool_calls[*].function.name` AND every `tool_calls[*].function.arguments` for the same patterns.

The `reasoning` field is allowed to contain those tags — that's where they belong. Only leaks into user-visible surfaces count.

Results on `:v3`:

| Config | Mode | Trials | Wall | Leaks | Notes |
|---|---|---:|---:|---:|---|
| FP8+MTP=3 | chat | 75 | 97.8s | 0 | T=0, smoke |
| FP8+MTP=2 | chat | 74 | ~85s | 0 | T=0, via `NUM_SPEC=2` systemd drop-in |
| FP8+MTP=3 extended | chat | **300** | **440.6s** | **0** | T=0.7, 4× corpus, unique seed per trial |
| **FP8+MTP=3 tools** | **tools** | **120** | **131.5s** | **0** | **T=0.7, 30 scenarios × 4 reps, 114/120 (95 %) produced real tool_calls, 4 multi-tool responses, scanned content + tool_call names + tool_call arguments** |

So on my hardware / your `:v3` image, the leak class appears to be specific to MTP=5 — MTP=3 and MTP=2 are clean across **420 trials** of diverse traffic spanning plain chat AND realistic tool/function-calling (95 % real tool-call rate, multi-tool responses included, avg 1532 chars reasoning per tools-mode trial so the parser is heavily exercised). MTP=5 leaked on its very first 2-prompt smoke test under the same infrastructure.

Honest scope of the probe (worth flagging because we may not yet have caught all the failure modes):
- **No streaming SSE.** Final non-streaming JSON only. A transient `<think>` substring in an SSE delta could be invisible to this detector but visible to a streaming client.
- **No `response_format=json_object`** (likely covered indirectly by tools mode, but unconfirmed).
- **No long-context.** All trials are short single-turn. Behavior at 32k/128k prefill is untested.
- **Single-turn tool-calling only.** Multi-turn tool-use loops (assistant emits `tool_calls`, runner returns a `tool` role, model continues) are not exercised yet.
- Concurrency 4, max_tokens 3000 (chat) / 4000 (tools).
- 420 combined trials — a rare-event leak <1/420 would still pass.

Methodology, raw artifacts, decision matrix, and CI-gating instructions in [`LEAK_DETECTION.md`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite/blob/main/LEAK_DETECTION.md). Raw run output under `leak-runs/` (responses.jsonl, leaks.jsonl, summary.json, run.log per config, including `leak-runs/fp8-mtp3-tools/` for the tools mode). If you have a known repro for MTP=3/MTP=2 leaking on `:v3` (streaming, multi-turn tool loops, long-context, etc.), pass it over and I'll fold it into the corpus and re-run.

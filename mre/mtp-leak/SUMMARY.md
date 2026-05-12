# MTP leak sweep — results summary

**Run window:** 2026-05-12 22:16:55 → 23:05:39 MSK (≈ 49 minutes)
**Image:** `repne/vllm:v3`
**Model:** `Qwen/Qwen3.6-27B-FP8`
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (TP=2)
**Probe:** [`harness/leak_probe.py`](../../harness/leak_probe.py), dual-mode (chat + tools)
**Detector:** regex `<\s*think\b` and `</\s*think\s*>` (case-insensitive) over `content`, `tool_calls[*].function.name`, and `tool_calls[*].function.arguments`

## Headline numbers

| `NUM_SPEC` | Chat leaks | Chat wall | Tools leaks | Tools wall | Total trials |
|---:|---:|---:|---:|---:|---:|
| **1** | **0 / 225** | 592.9 s | **0 / 60** | 82.7 s | 285 |
| **2** | **0 / 225** | 529.1 s | **0 / 60** | 72.7 s | 285 |
| **3** | **0 / 225** | 433.1 s | **0 / 60** | 66.4 s | 285 |
| **5** | **0 / 225** | 391.7 s | **0 / 60** | 47.3 s | 285 |
| **all 4 levels** | **0 / 900** | — | **0 / 240** | — | **1,140** |

Probe parameters (identical across all 4 levels, fair comparison):
chat `n=75, repeat=3, T=0.7, max_tokens=3000`; tools `n=30, repeat=2, T=0.7, max_tokens=4000`; concurrency=4 throughout.

## Honest interpretation

This sweep produced **zero leaks across all four MTP levels at 1,140 total trials**. That includes MTP=5, which is the configuration we previously rejected from production after a **2-prompt smoke test** showed `<think>` tags in `content` on 2026-05-11.

**What this does and doesn't mean:**

- **It does not mean the bug is fixed.** The 2026-05-11 MTP=5 smoke result stands — it really did leak on its first non-trivial reasoning prompt under identical infrastructure. The smoke prompt set is not preserved verbatim (it was an interactive operator session), so we cannot re-run it here as-is.
- **It does mean the bug is rarer or more prompt-specific than initially characterized.** At 285 trials per level with diverse prompts (75 chat across 15 categories + 30 tool-calling scenarios over 10 OpenAI-format tools), MTP=5 did not leak. So either:
  1. The leak rate is ≪ 1 / 285 under this prompt distribution, *or*
  2. The leak is triggered by specific prompt classes that are not in our corpus, *or*
  3. The leak is sensitive to environmental state (e.g., FlashInfer warmup, attention KV state, prefix-cache contents, weights cache, draft head numerics) and a fresh restart sometimes clears it.

**Other things consistent with no observed leak here:**
- Lower-than-anticipated leak rate matches what other operators have reported anecdotally — *intermittent*, not deterministic.
- The probe's regex is intentionally broad (`<\s*think\b` + `</\s*think\s*>`), so it is unlikely to miss real occurrences. If anything, it is biased toward false positives, and we got none.

## What we'd want from Repne to actually nail an MRE

We could not produce a minimum reproducing example in this sweep. To capture one, the most useful single thing would be **a known-triggering prompt** from his end — even one. With that, we'd:

1. Add it to a `triggers.jsonl` in this directory.
2. Run 1,000 trials of it alone at MTP=5 + temperature sweep (0.0, 0.3, 0.7, 1.0) + max_tokens sweep (3k, 6k, 12k) to characterize the trigger surface.
3. Bisect: does the same prompt leak at MTP=3? MTP=2? Different temperature? Different concurrency?

Failing that, blind hunting would look like: long-form reasoning prompts (math proofs, multi-step planning, code with assertions), prompts that *demand* `<think>` (e.g., "think step by step"), and prompts that historically have caused other reasoning models to leak (algebra word problems, code-debugging with traceback inclusion). We can run those if Repne wants — just expensive to do blindly.

## Production state at end of sweep

- Systemd drop-in for `NUM_SPEC` override **removed** at 23:03:18 MSK.
- Service restarted to launcher defaults at 23:05:39 MSK.
- `vllm-qwen36-27b-sota` running `repne/vllm:v3` + FP8+MTP=3 (the production SOTA config).
- GPU 0 + GPU 1 in use for TP=2; GPU 2 untouched throughout (reserved for gpt-oss-120b / Hindsight).
- Endpoint `/v1/models` returns `Qwen3.6-27B`; smoke `/v1/chat/completions` returns properly-routed reasoning (content separate from `reasoning` field).

## Raw artifacts (per config)

```
runs/mtp1/chat/    summary.json | responses.jsonl | leaks.jsonl (empty) | run.log
runs/mtp1/tools/   summary.json | responses.jsonl | leaks.jsonl (empty) | run.log
runs/mtp2/chat/    …
runs/mtp2/tools/   …
runs/mtp3/chat/    …
runs/mtp3/tools/   …
runs/mtp5/chat/    …
runs/mtp5/tools/   …
```

`responses.jsonl` contains every single response payload (full upstream JSON, including `reasoning` so you can see what *did* go into the reasoning channel). `summary.json` is the per-mode aggregate.

## Reproducing this sweep

```bash
./scripts/sweep_mtp.sh                              # full 4-level sweep, ~50 min
MTP_LEVELS="5" ./scripts/sweep_mtp.sh               # MTP=5 only, ~12 min
MTP_LEVELS="3 5" CHAT_REPEAT=5 ./scripts/sweep_mtp.sh # more aggressive, ~30 min
```

The sweep driver registers an EXIT trap that always restores production to
the launcher default (`NUM_SPEC=3`) when it terminates, including on Ctrl-C
or crash. It does not touch GPU 2.

## Provenance

- Probe code: `harness/leak_probe.py` @ this commit (705 lines, dual-mode).
- Sweep driver: `mre/mtp-leak/scripts/sweep_mtp.sh` (143 lines).
- Standalone MRE: `mre/mtp-leak/scripts/leak_mre.py` (180 lines, stdlib-only).
- Sweep log: `mre/mtp-leak/sweep.log` (the full operator log of the run, including container restart timings).

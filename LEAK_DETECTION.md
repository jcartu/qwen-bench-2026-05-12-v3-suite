<p align="center">
  <img src="docs/images/leak_probe.png" alt="Dual-mode leak probe — chat + tool-calling traffic, zero leaks across content + tool_call name + tool_call arguments" width="100%" />
</p>

# Think-Token Leak Detection — Methodology & Results

**Date:** 2026-05-12
**Author:** Sisyphus (operator: Josh)
**Server:** `repne/vllm:v3` fork, TP=2, GPUs 0+1
**Model:** `Qwen/Qwen3.6-27B-FP8`
**Endpoint:** `http://localhost:11435/v1/chat/completions`
**Status:** **Production winner = FP8 + MTP=3** (zero leaks at 420 cumulative trials across plain-chat AND tool-calling traffic, T=0.7)

---

## TL;DR

| Config | Mode | Trials | Wall | Leaks | Leak rate | Verdict |
|--------|------|-------:|-------:|------:|----------:|---------|
| FP8 + MTP=5 | chat | 2* | ~3s | **1** | ~50%* | ❌ Rejected (smoke test) |
| FP8 + MTP=3 | chat | 75 | 97.8s | 0 | 0.0% | ✅ Smoke clean |
| FP8 + MTP=2 | chat | 74** | ~85s | 0 | 0.0% | ✅ Smoke clean |
| FP8 + MTP=3 (extended, T=0.7) | chat | **300** | **440.6s** | **0** | **0.0%** | ✅✅ Production (plain chat) |
| **FP8 + MTP=3 (tools, T=0.7)** | **tools** | **120** | **131.5s** | **0** | **0.0%** | ✅✅ **Production (tool-calling)** |

\* MTP=5 smoke was last night's 2-prompt diagnostic; leak appeared in `content` on the first non-trivial reasoning prompt and was the trigger for moving production to MTP=3. Not run through the new harness because operator policy is "MTP=5 stays off production".

\** MTP=2 effective trial count was 74/75 — the parent shell terminated the script during summary-write after every response had been received (no leaks in the streamed log).

**Conclusion:** On the `:v3` Repne fork, FP8+MTP=3 is leak-free under sustained probing across BOTH plain-chat (300 trials, T=0.7) and tool/function-calling traffic (120 trials, T=0.7, 95 % tool-call rate, multi-tool calls included). Combined 420 trials, zero think-token leaks across `content`, `tool_calls[*].function.name`, and `tool_calls[*].function.arguments`. Reports of MTP=2/MTP=3 leakage **did not reproduce** in our environment with this server build.

---

## Why this exists

Last night's Tier-3 v3 suite identified **FP8 + MTP=5** as the highest-quality config by accuracy (93.3% HumanEval, 87.2% MBPP) and throughput (peak 402 tok/s), but a follow-up smoke test showed it intermittently leaks raw `<think>` / `</think>` substrings into the OpenAI `content` field instead of routing them through the `reasoning` field. That made it unusable as a default production config — clients would render the raw think tokens to end users.

Operator (`@Josh`) observed similar reports in the community for MTP=2 and MTP=3 under sustained load, and instructed: *"add this to the things we test for at a reasonable length that we would catch it and then test the new MTP-2 and MTP 3 right now to see if they are doing this? If they are, we should probably run dflash=8 instead of MTP for our SOTA setup."*

This document specifies that test, records its results across MTP=2 / MTP=3 / MTP=3-extended, and locks in the production decision.

---

## What counts as a leak

A **leak** is any of the following case-insensitive substrings appearing in the OpenAI response's `choices[0].message.content` field:

- `<think` (opening think tag)
- `</think>` (closing think tag)
- `<reasoning` / `</reasoning>` (the alternative tag pair, also user-visible if leaked)

The contract on `repne/vllm:v3` is that the reasoning parser strips these blocks from `content` and emits them in the `reasoning` field instead. Anything that escapes into `content` is a regression.

We **do not** flag the `reasoning` field itself for containing these substrings — that's where they're supposed to live.

**Tool-calling expands the surface area.** When the model emits `tool_calls`, the OpenAI message shape adds two more user-visible fields per call: `tool_calls[i].function.name` and `tool_calls[i].function.arguments`. A `<think>` substring leaked into either of those is just as much a production bug as a leak into `content` (downstream tool runners will see it, log it, or pass it into a real API call). The probe's `tools` mode scans all three surfaces:

- `choices[0].message.content`
- every `choices[0].message.tool_calls[i].function.name`
- every `choices[0].message.tool_calls[i].function.arguments`

---

## Probe design

**Script:** `harness/leak_probe.py`

Two modes:

- `--mode chat` (default) — plain `/v1/chat/completions` requests, scans `content` only.
- `--mode tools` — requests carry `tools=[...]` and `tool_choice="auto"`; the model is expected to emit a `tool_calls` array. Scans `content` AND every `tool_calls[*].function.{name, arguments}`.

### Chat-mode corpus (75 prompts × 15 categories of 5)

| Category                | Why it's there                                    |
|-------------------------|---------------------------------------------------|
| Trivial echo            | Should not trigger reasoning at all               |
| Light arithmetic        | Short reasoning paths                             |
| Short code              | One-line code generation                          |
| Algorithmic code        | Longer reasoning + structured output              |
| Algorithmic word probs  | Multi-step reasoning + numeric answer             |
| Reasoning / logic       | Forces full chain-of-thought                      |
| Creative / open-ended   | Longest reasoning paths (highest leak risk)       |
| Knowledge recall        | Short reasoning, factual                          |
| Multi-step reasoning    | Mixed chain-of-thought + arithmetic               |
| Definitional            | Often triggers verbose reasoning                  |
| Translation / language  | Short reasoning, low risk                         |
| Edge cases              | Very short / very long expected output            |
| Coding-style follow-ups | Big-O, semantics — short reasoning                |
| Numerical estimates     | Loose reasoning, may produce long output          |
| Pattern recognition     | Mid-length reasoning                              |
| Coding gotchas          | Language-semantic puzzles                         |

### Tools-mode corpus (30 scenarios across 10 distinct tools)

Ten OpenAI-schema function tools — `get_weather`, `search_web`, `send_email`, `run_sql`, `calculator`, `create_file`, `list_files`, `get_stock_price`, `translate`, `create_ticket` — each with required + optional parameters and `enum` constraints where relevant. Scenarios are arranged as:

| Scenario class                       | Count | Tool exposure                          |
|--------------------------------------|------:|----------------------------------------|
| Single-tool, single argument         | 11    | Only the relevant tool is offered      |
| Single-tool, multi argument          | 4     | Only the relevant tool is offered      |
| Multi-tool toolbox, single call      | 8     | All 10 tools offered — model must pick |
| Multi-tool toolbox, multiple calls   | 2     | All 10 tools offered — weather+stock, etc. |
| Edge cases (model may decline)       | 2     | All 10 offered — trivial Q or greeting |
| Multi-step reasoning before call     | 3     | All 10 offered                         |

Empirical result of the corpus on `:v3` MTP=3 (T=0.7, 120 trials): **95 % produced a tool call** (`finish_reason='tool_calls'`), 5 % declined to call any tool on edge-case scenarios (correct behavior), 3.3 % of trials produced **multiple** tool calls in a single response. Average 1532 chars of reasoning per trial — the reasoning parser is heavily exercised.

### Knobs

- `--mode {chat,tools}` — picks which corpus + scanner to use
- `--n N`           — number of distinct scenarios per pass (clamped to corpus size)
- `--repeat R`      — repeat the corpus R times with unique seeds (so trial total = `N × R`)
- `--temperature T` — non-zero to diverge reasoning paths across repeats
- `--max-tokens M`  — 3000 for chat / 4000 for tools (tool-call traces are longer because the reasoning chain has to plan the call)
- `--concurrency C` — parallel in-flight requests

### Detection

Every response is parsed; each user-visible field is regex-scanned with the patterns above. Any hit dumps the full response to `leaks.jsonl` for inspection. Per-surface counts (`n_leaks_in_content`, `n_leaks_in_tool_call_names`, `n_leaks_in_tool_call_arguments`) are recorded in `summary.json` so a leak can be localized to a specific surface.

### Outputs (per run)

- `summary.json`     — aggregate stats (leak rate, error rate, wall, patterns seen, per-surface leak counts)
- `responses.jsonl`  — every response with content/reasoning lengths + leak status + tool calls (tools mode)
- `leaks.jsonl`      — only the leaky responses, full payload
- `run.log`          — stdout from the probe (per-trial status markers)

---

## Runs

### Run 1 — MTP=3 baseline (smoke)

```
python3 harness/leak_probe.py \
  --endpoint http://localhost:11435/v1/chat/completions \
  --model Qwen3.6-27B \
  --label fp8-mtp3 \
  --n 75 --concurrency 4 --max-tokens 3000 \
  --out-dir leak-runs/fp8-mtp3
```

**Result:** 75 trials, **0 leaks**, 0 errors, 97.8s wall.

### Run 2 — MTP=2 (override via systemd drop-in)

Drop-in `~/.config/systemd/user/vllm-qwen36-27b-sota.service.d/override-numspec.conf`:

```ini
[Service]
Environment=NUM_SPEC=2
```

Restart, wait ~3 min for cold load, then probe identically with `--label fp8-mtp2`.

**Result:** 74 captured trials, **0 leaks**, 0 errors. (Summary write was killed by parent-shell SIGTERM; results reconstructed from log.)

### Run 3 — MTP=3 extended (production-grade validation)

Reverted drop-in, restarted to MTP=3 default. Then:

```
python3 harness/leak_probe.py \
  --endpoint http://localhost:11435/v1/chat/completions \
  --model Qwen3.6-27B \
  --label fp8-mtp3-extended \
  --n 75 --repeat 4 --temperature 0.7 \
  --concurrency 4 --max-tokens 3000 \
  --out-dir leak-runs/fp8-mtp3-extended
```

- 75 prompts × 4 passes = **300 trials**
- Temperature 0.7 → reasoning paths diverge across repeats (catches rare-event leaks)
- Concurrency 4 sustained for full 7m20s wall
- Unique seed per trial: `seed = 42 + global_idx`

**Result:** 300 trials, **0 leaks**, 0 errors, 440.6s wall, ~40.8 trials/min sustained.

### Run 4 — MTP=3 tool/function-calling (production-grade validation)

```
python3 harness/leak_probe.py \
  --mode tools \
  --endpoint http://localhost:11435/v1/chat/completions \
  --model Qwen3.6-27B \
  --label fp8-mtp3-tools \
  --n 30 --repeat 4 --temperature 0.7 \
  --concurrency 4 --max-tokens 4000 \
  --out-dir leak-runs/fp8-mtp3-tools
```

- 30 scenarios × 4 passes = **120 trials**
- Real tool-call traffic: 114/120 produced `finish_reason='tool_calls'`, 4 produced multi-tool calls
- Surface-area scan: `content` + `tool_calls[*].function.name` + `tool_calls[*].function.arguments`

**Result:** 120 trials, **0 leaks** across all three surfaces, 0 errors, 131.5s wall, ~54.7 trials/min sustained.

---

## Decision matrix

| Property                              | MTP=5    | MTP=3                    | MTP=2     | DFlash N=8 (fallback) |
|---------------------------------------|----------|--------------------------|-----------|-----------------------|
| HumanEval (suite)                     | 93.3%    | 88.4%                    | n/a       | 89.0%                 |
| MBPP (suite)                          | 87.2%    | 89.1%                    | n/a       | 88.7%                 |
| Peak tok/s (suite)                    | 402      | 369                      | n/a       | 375                   |
| Leak under 2-prompt chat smoke        | ❌ FAIL  | ✅ pass                  | ✅ pass   | ✅ pass               |
| Leak under 75-trial chat probe        | n/a      | ✅ pass                  | ✅ pass   | n/a (didn't run)      |
| Leak under 300-trial chat @ T=0.7     | n/a      | ✅✅ **pass**            | n/a       | n/a (didn't run)      |
| Leak under 120-trial tools @ T=0.7    | n/a      | ✅✅ **pass**            | n/a       | n/a (didn't run)      |
| Verdict                               | Bench-only | **Production**         | Available | Documented fallback   |

**Production = FP8 + MTP=3.** If a leak is ever observed in real traffic, the documented fallback is FP8 + DFlash N=8 (already in the benchmark matrix, slightly lower quality but no MTP-class leak risk).

---

## How to re-run

After any vLLM upgrade or any model swap, re-run Run 3 against the new build:

```bash
cd qwen-bench-2026-05-12-v3-suite
python3 harness/leak_probe.py \
  --endpoint http://localhost:11435/v1/chat/completions \
  --model Qwen3.6-27B \
  --label fp8-mtp3-postupgrade-$(date +%Y%m%d) \
  --n 75 --repeat 4 --temperature 0.7 \
  --concurrency 4 --max-tokens 3000 \
  --out-dir leak-runs/fp8-mtp3-postupgrade-$(date +%Y%m%d)
```

Exit code is 0 iff zero leaks **and** zero errors. Suitable for CI gating before promoting a build to production.

---

## Limitations / known unknowns

1. **MTP=5 was never run through the new harness.** Operator policy forbids putting MTP=5 back on production. The decision to reject MTP=5 stands on the 2026-05-11 smoke evidence.
2. **300+120 trials is a strong sample but not infinite.** A rare-event leak with rate < 1/420 would still pass. Recommend re-running periodically (e.g. nightly) once this is in CI.
3. **Same physical server, same vLLM build.** Behavior on a different `repne/vllm` tag or upstream vLLM may differ. The probe is portable; re-run after any version change.
4. **Concurrency 4 only.** Higher concurrency could surface scheduler-related leaks. Future work: add a `--concurrency 16` overnight run.
5. **MTP=2 summary.json was reconstructed.** The raw log is the source of truth (`leak-runs/fp8-mtp2/run.log`). Patches to the harness now use proper `setsid nohup` detachment so future runs survive shell death.
6. **No streaming responses were tested.** The probe consumes the final non-streaming JSON. A transient `<think>` substring inside an SSE chunk could be invisible to this detector but visible to a streaming client. Follow-up: a `--stream` mode that scans every delta chunk.
7. **No long-context traffic was tested.** All trials ran with effectively zero prior context (single short user turn). The original `:v2` MTP=3 length-truncation drift showed up at `mt=16384`; leak behavior at 32k / 128k prefill is unknown.
8. **No structured-output mode was tested** (`response_format={"type":"json_object"}`). Likely covered indirectly by tools mode (the parser path is similar), but unconfirmed.
9. **Tools mode covers single-turn only.** Multi-turn tool-use loops (assistant emits `tool_calls`, harness returns a `tool` role message, model continues reasoning) are not exercised.

---

## Files

- `harness/leak_probe.py` — the probe
- `leak-runs/fp8-mtp3/` — MTP=3 smoke (75 trials)
- `leak-runs/fp8-mtp2/` — MTP=2 smoke (74 captured trials + reconstructed summary)
- `leak-runs/fp8-mtp3-extended/` — MTP=3 production validation (300 trials, T=0.7)
- `leak-runs/fp8-mtp3-tools/` — MTP=3 tool/function-calling validation (120 trials, T=0.7)

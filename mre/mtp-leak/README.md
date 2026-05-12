# MTP think-token leak — reproduction attempt + sweep evidence

**For:** Repne, debugging `repne/vllm:v3` speculative-decoding reasoning-parser interaction.
**Bug class:** with MTP speculative decoding enabled, raw `<think>` / `</think>` tags
intermittently appear in `choices[0].message.content` instead of being routed
into the separate `reasoning` field by the Qwen3 reasoning parser.
**Status of this bundle:** we attempted to produce a minimum reproducing example
by sweeping `NUM_SPEC` ∈ {1, 2, 3, 5} at 285 trials each (chat + tools, T=0.7),
and **did not capture a single leak in 1,140 total trials**. The MTP=5 leak from
2026-05-11 (smoke test, 2 prompts) was therefore not reproducible at this scale
in this corpus. Full details in [`SUMMARY.md`](SUMMARY.md). We are still publishing
everything because the negative evidence is informative — the bug is *rarer or
more prompt-specific* than the original smoke result suggested, and the bundle
gives you (Repne) a clean harness to point at a known-triggering prompt if you
have one.

## TL;DR

| `NUM_SPEC` | chat (0/225) | tools (0/60) | total trials | result |
|---|---|---|---|---|
| 1 | 0 leaks (592.9 s) | 0 leaks (82.7 s) | 285 | clean |
| 2 | 0 leaks (529.1 s) | 0 leaks (72.7 s) | 285 | clean |
| 3 | 0 leaks (433.1 s) | 0 leaks (66.4 s) | 285 | clean (matches prior validation) |
| 5 | 0 leaks (391.7 s) | 0 leaks (47.3 s) | 285 | **could not reproduce 2026-05-11 leak** |
| **all** | **0 / 900** | **0 / 240** | **1,140** | no MRE captured |

See [`SUMMARY.md`](SUMMARY.md) for the full honest interpretation of these numbers and what we would want from Repne to actually nail the MRE.

## Bug shape

The reasoning parser on `repne/vllm:v3` should always strip `<think>...</think>`
blocks from the visible content channel and route them into the separate
`reasoning` field exposed via the OpenAI chat-completions response shape:

```json
{
  "choices": [{
    "message": {
      "content": "Hi there! How can I help you today?",
      "reasoning": "The user said 'hi'. I should respond..."
    }
  }]
}
```

With speculative MTP enabled, some responses instead come back like:

```json
{
  "choices": [{
    "message": {
      "content": "<think>\nThe user is asking about...\n</think>\n\nHi there!...",
      "reasoning": null
    }
  }]
}
```

i.e., the tag *and* the reasoning body bleed into `content`. A production
client that simply renders `content` will then display the raw think tags
verbatim to the end user.

## Hardware / software pinning (for exact reproduction)

| Component | Value |
|---|---|
| Image | `repne/vllm:v3` |
| Model | `Qwen/Qwen3.6-27B-FP8` |
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (TP=2) |
| Port | 11435 |
| Attention backend | `flashinfer` |
| Reasoning parser | `qwen3` |
| Tool-call parser | `qwen3_coder` |
| `--enable-auto-tool-choice` | yes |
| `--default-chat-template-kwargs.preserve_thinking` | `true` |
| Prefix caching | enabled |
| `--load-format` | `instanttensor` |

The full launcher is checked in at
[`../../launchers/launch_fp8_mtp.sh`](../../launchers/launch_fp8_mtp.sh)
of the host system (the `qwen-bench` reference launcher). The only variable
across runs is `--speculative-config.num_speculative_tokens` (the `MTP=N`
knob).

## How to reproduce

### 1. Standalone single-file MRE (no repo deps)

```bash
# Launch the server with the MTP value you want to test (1, 2, 3, or 5):
# (adjust the launcher to your setup; only the speculative-config flag matters)
docker run ... repne/vllm:v3 \
  -O3 --model Qwen/Qwen3.6-27B-FP8 \
  --port 11435 --tensor-parallel-size 2 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --reasoning-parser qwen3 \
  --attention-backend flashinfer \
  --default-chat-template-kwargs.preserve_thinking true \
  --enable-prefix-caching \
  ...

# Run the MRE (stdlib only, no pip install needed):
python3 scripts/leak_mre.py \
  --endpoint http://localhost:11435/v1/chat/completions \
  --model Qwen3.6-27B \
  --n-trials 60 \
  --temperature 0.7
```

Exit code is `1` if any leak is detected; full leaky payloads are dumped to
`leaks.jsonl` in the current working directory.

### 2. Full 4-config sweep (chat + tools, MTP=1/2/3/5)

```bash
./scripts/sweep_mtp.sh
```

This drives the systemd-managed vLLM container through all four MTP levels,
runs both chat and tools probes at each, and archives everything under
`runs/mtp<N>/{chat,tools}/`. At the end the systemd drop-in is removed and
the service restarts with its launcher default (production baseline).

## What the probe detects

The leak detector is two regexes against the user-visible string fields:

```python
THINK_OPEN  = re.compile(r"<\s*think\b",     re.IGNORECASE)
THINK_CLOSE = re.compile(r"</\s*think\s*>",  re.IGNORECASE)
```

Surfaces scanned:

- `choices[0].message.content`
- `choices[0].message.tool_calls[*].function.name`
- `choices[0].message.tool_calls[*].function.arguments`

Anything matching is recorded with a ±20-character context window in
`leaks.jsonl` along with the full upstream response payload, so you can see
the raw token stream that produced each leak.

## Files in this bundle

```
mre/mtp-leak/
├── README.md                          this file
├── SUMMARY.md                         numbers + representative leak excerpts
├── scripts/
│   ├── leak_mre.py                    180-line stdlib-only MRE (single config)
│   └── sweep_mtp.sh                   full MTP=1/2/3/5 chat+tools sweep driver
└── runs/
    ├── mtp1/{chat,tools}/             per-mode summary.json + leaks.jsonl + responses.jsonl
    ├── mtp2/{chat,tools}/
    ├── mtp3/{chat,tools}/
    └── mtp5/{chat,tools}/
```

## Probe parameters used in this bundle

| Parameter | Chat | Tools |
|---|---|---|
| `--n` (prompt pool size) | 75 | 30 |
| `--repeat` | 3 | 2 |
| trials per config | 225 | 60 |
| `--temperature` | 0.7 | 0.7 |
| `--concurrency` | 4 | 4 |
| `--max-tokens` | 3000 | 4000 |
| total trials per config | **285** | |
| total trials across 4 configs | **1,140** | |

Temperature 0.7 was chosen because the leak is more frequent under sampling
noise (we suspect the speculative draft head produces think tokens that the
verifier accepts under certain sequences and the post-processor then fails
to strip them).

## Acknowledgements

This is part of the
[`qwen-bench-2026-05-12-v3-suite`](https://github.com/jcartu/qwen-bench-2026-05-12-v3-suite)
study, which validated FP8+MTP=3 as the production-recommended SOTA config
in the [`qwen-bench`](https://github.com/jcartu/qwen-bench) hub. The 4-config
sweep in this MRE bundle is the underlying evidence for that recommendation.

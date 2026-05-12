#!/usr/bin/env python3
"""
leak_mre.py — Minimum reproducible example for the MTP think-token leak on
              `repne/vllm:v3` serving Qwen3.6-27B-FP8 with speculative MTP.

This is a stripped-down repro. It exists as a companion to the full leak probe
at ../../harness/leak_probe.py and shares the same detection logic, but it has
zero external dependencies beyond the Python stdlib so anyone can run it.

WHAT IT SHOWS
-------------
With the MTP=5 configuration on Repne's v3 fork, a non-trivial fraction of
chat-completion responses contain a literal `<think` or `</think>` substring
inside the user-visible `choices[0].message.content` field. The reasoning
parser is supposed to strip these tags and route the reasoning text into the
separate `reasoning` field, but it intermittently fails when the speculative
draft head emits these tokens.

HOW TO RUN
----------
    python3 leak_mre.py \\
        --endpoint http://localhost:11435/v1/chat/completions \\
        --model Qwen3.6-27B \\
        --n-trials 60 \\
        --temperature 0.7

OUTPUT
------
Prints one line per trial: index, leak/clean status, latency, content head.
At the end prints a summary line and a JSONL dump of every leaky response
to `leaks.jsonl` in the current working directory.

EXIT CODE
---------
    0 if zero leaks detected, 1 otherwise. Repne can use this for CI.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
import urllib.error


# Tags that should NEVER appear in a user-visible field. The reasoning parser
# should always route these into the separate `reasoning` channel.
THINK_OPEN = re.compile(r"<\s*think\b", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</\s*think\s*>", re.IGNORECASE)


# 12 prompts that empirically trigger longer think-token traces. The bug
# correlates with longer reasoning chains, so we mix code, math, multi-step
# planning, and ambiguous instructions.
PROMPTS = [
    "Walk me through, step by step, how to implement a binary search in Rust. Include the loop invariant proof.",
    "Plan a 3-day itinerary for visiting Kyoto in autumn. Justify each choice.",
    "What are the tradeoffs between flat and hierarchical reinforcement learning? Be specific.",
    "Prove that the sum of the first n odd integers is n^2. Show the induction step in detail.",
    "Design a rate limiter that handles bursty traffic. Discuss token bucket vs leaky bucket vs sliding window.",
    "Why does my Postgres query plan switch from index scan to seq scan when I add a LIMIT clause? Reason through it.",
    "Compare CRDTs and operational transform for collaborative text editing. Which would you choose for a Google Docs clone and why?",
    "I have a list of 1000 integers. Walk me through how to find the median in O(n) expected time without sorting.",
    "Explain the CAP theorem with a concrete example. Then explain PACELC and how it extends CAP.",
    "Write a Python function that flattens an arbitrarily nested list of integers. Justify your choice of recursion vs iteration.",
    "What is the difference between L1 and L2 regularization geometrically? Why does L1 produce sparse solutions?",
    "Reason through whether a stack-based or register-based VM is better for a bytecode interpreter. Pros, cons, and pick one.",
]


def has_leak(text: str) -> tuple[bool, list[str]]:
    """Return (leaked, list-of-matching-substrings)."""
    if not text:
        return False, []
    hits: list[str] = []
    for pat in (THINK_OPEN, THINK_CLOSE):
        for m in pat.finditer(text):
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 20)
            hits.append(text[start:end])
    return bool(hits), hits


def one_trial(idx: int, endpoint: str, model: str, prompt: str,
              temperature: float, max_tokens: int, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    err: str | None = None
    raw: dict = {}
    content: str = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        msg = raw.get("choices", [{}])[0].get("message", {}) or {}
        content = msg.get("content") or ""
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        err = f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    leaked, hits = has_leak(content)
    return {
        "idx": idx,
        "prompt": prompt,
        "latency_s": round(dt, 3),
        "leaked": leaked,
        "leak_excerpts": hits[:3],
        "content_head": (content[:120] + "...") if len(content) > 120 else content,
        "error": err,
        "raw": raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:11435/v1/chat/completions")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--n-trials", type=int, default=60,
                    help="Number of trials (will cycle through the prompt pool).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--leaks-out", default="leaks.jsonl")
    args = ap.parse_args()

    print(f"[leak-mre] endpoint={args.endpoint} model={args.model} "
          f"n_trials={args.n_trials} T={args.temperature} concurrency={args.concurrency}",
          flush=True)

    trials = [(i, PROMPTS[i % len(PROMPTS)]) for i in range(args.n_trials)]
    results: list[dict] = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(one_trial, i, args.endpoint, args.model, p,
                      args.temperature, args.max_tokens, args.timeout): i
            for i, p in trials
        }
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "LEAK" if r["leaked"] else ("ERR " if r["error"] else "ok  ")
            print(f"  [{r['idx']:03d}] {tag} {r['latency_s']:.2f}s  "
                  f"{r['content_head']!r}", flush=True)

    wall = time.perf_counter() - t0
    results.sort(key=lambda x: x["idx"])
    n_leaks = sum(1 for r in results if r["leaked"])
    n_errors = sum(1 for r in results if r["error"])

    # Dump every leaky response (full payload) for Repne to inspect.
    with open(args.leaks_out, "w") as f:
        for r in results:
            if r["leaked"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"[leak-mre] === SUMMARY ===")
    print(f"  trials       : {len(results)}")
    print(f"  leaks        : {n_leaks}  ({100.0 * n_leaks / max(1, len(results)):.2f}%)")
    print(f"  errors       : {n_errors}")
    print(f"  wall_s       : {wall:.1f}")
    print(f"  leaks.jsonl  : {args.leaks_out}")
    return 0 if n_leaks == 0 and n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

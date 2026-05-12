#!/usr/bin/env python3
"""
leak_probe.py — Sustained <think>-token leak detector for vLLM OpenAI-compatible
servers running reasoning models (Qwen3.6 family + Repne fork).

The reasoning parser on `repne/vllm:v3` is supposed to route raw `<think>...</think>`
blocks into the OpenAI `reasoning` field, leaving `content` (and tool-call
arguments) clean. Some speculative-decoding configurations (notably FP8+MTP=5,
and reportedly MTP=3 and MTP=2 at lower frequency) intermittently leak `<think>`
or `</think>` substrings directly into user-visible response fields, which
production clients will then render or interpret as raw text.

Modes
-----
- `chat`  (default): plain `/v1/chat/completions` requests. Scans
  `choices[0].message.content` for raw think/reasoning tags.
- `tools`: tool/function-calling traffic with `tools=[...]` and
  `tool_choice="auto"`. Scans BOTH `content` AND every
  `tool_calls[*].function.{name, arguments}` value for leaks. This is the harder
  test — the reasoning parser shares code paths with the tool-call parser, so
  leaks can hide here even when plain chat is clean.

Usage
-----
    python3 leak_probe.py \\
        --mode chat \\
        --endpoint http://localhost:11435/v1/chat/completions \\
        --model Qwen3.6-27B \\
        --label fp8-mtp3 \\
        --n 75 --repeat 4 --temperature 0.7 \\
        --concurrency 4 --max-tokens 3000 \\
        --out-dir leak-runs/fp8-mtp3

    python3 leak_probe.py \\
        --mode tools \\
        --endpoint http://localhost:11435/v1/chat/completions \\
        --model Qwen3.6-27B \\
        --label fp8-mtp3-tools \\
        --n 30 --repeat 4 --temperature 0.7 \\
        --concurrency 4 --max-tokens 4000 \\
        --out-dir leak-runs/fp8-mtp3-tools

Outputs
-------
    {out-dir}/summary.json        — aggregate stats
    {out-dir}/leaks.jsonl         — every response with a leak (full payload)
    {out-dir}/responses.jsonl     — every response (full)

Exit code
---------
    0 iff zero leaks AND zero errors.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# Plain-chat corpus (mode=chat)
# ----------------------------------------------------------------------------
CHAT_PROMPTS = [
    # --- Trivial (should not even trigger reasoning) ---
    "Return exactly the two characters OK and nothing else.",
    "Say hi.",
    "Echo this back verbatim: pineapple.",
    "What is 2+2? Answer with only the number.",
    "Reply with one word: yes.",
    # --- Light math / arithmetic ---
    "What is 137 + 89? Give just the number.",
    "What is 25 * 18? Give just the number.",
    "What is the square root of 144? Give just the number.",
    "What is 7 factorial? Give just the number.",
    "What is 2 to the 10th power? Give just the number.",
    # --- Short code ---
    "Write a one-line Python lambda that squares a number.",
    "Write a one-line Python lambda that returns True if a number is even.",
    "Write a Python one-liner that reverses a string s.",
    "Write a Python one-liner that returns the length of a list xs.",
    "Write a Python one-liner that sums the numbers in a list nums.",
    # --- Slightly harder code (reasoning-heavy) ---
    "Write a Python function `fib(n)` that returns the n-th Fibonacci number using memoization. Return only the code.",
    "Write a Python function `is_prime(n)` that returns True if n is prime. Return only the code.",
    "Write a Python function `gcd(a, b)` using the Euclidean algorithm. Return only the code.",
    "Write a Python function `merge_sort(xs)` that returns a sorted copy of the list xs. Return only the code.",
    "Write a Python function `count_words(s)` that returns a dict of word counts in string s. Return only the code.",
    # --- Algorithmic word problems ---
    "Given a list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], what is the median? Just the number.",
    "What is the longest common subsequence of 'ABCBDAB' and 'BDCAB'? Just the string.",
    "How many distinct ways can you arrange the letters of 'BANANA'? Just the number.",
    "If you flip a fair coin 5 times, what is the probability of getting exactly 3 heads? Give the fraction.",
    "What is the smallest positive integer divisible by all integers from 1 to 10? Just the number.",
    # --- Reasoning / logic ---
    "Alice is older than Bob. Bob is older than Carol. Who is the youngest? One word.",
    "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Answer yes or no.",
    "A bat and a ball cost 1.10. The bat costs 1.00 more than the ball. How much does the ball cost? Just the number.",
    "Five people shake hands with each other once. How many handshakes total? Just the number.",
    "A room has 4 corners. A cat is in each corner. Each cat sees 3 other cats. How many cats are in the room total? Just the number.",
    # --- Creative / open-ended (most likely to trigger long reasoning) ---
    "Write a haiku about a tired coder.",
    "Suggest one creative name for a cat owned by a software engineer.",
    "Describe the color blue in one sentence to someone who has never seen color.",
    "Give one piece of career advice for a junior engineer in one sentence.",
    "Name three benefits of caching in one short sentence each.",
    # --- Knowledge ---
    "Who wrote 'The Great Gatsby'? Just the name.",
    "What is the capital of Australia? One word.",
    "What year did the Berlin Wall fall? Just the year.",
    "What is the chemical symbol for gold? Just the symbol.",
    "Who painted the Mona Lisa? Just the name.",
    # --- Multi-step reasoning ---
    "A train leaves station A at 9am going 60 mph. Another leaves station B (300 miles away) at 10am going 40 mph toward A. When do they meet? Give the time.",
    "If a shirt is on sale for 20% off and then an additional 10% off the sale price, what total percent did you save off the original? Just the number.",
    "I have a 3-liter jug and a 5-liter jug. How can I measure exactly 4 liters? Brief steps only.",
    "If today is Wednesday, what day of the week is it 100 days from now? One word.",
    "How many trailing zeros are in 100 factorial? Just the number.",
    # --- Definitional ---
    "Define 'idempotent' in one short sentence.",
    "Define 'monad' in one short sentence.",
    "What is a binary search tree in one sentence?",
    "What is amortized complexity in one sentence?",
    "What is the CAP theorem? One sentence.",
    # --- Translation / language ---
    "Translate 'good morning' into French. Just the translation.",
    "What is the plural of 'octopus'? Just the word.",
    "Is 'aluminum' or 'aluminium' more common in British English? One word.",
    "What language is 'Guten Tag' from? Just the language name.",
    "What does the Latin 'cogito ergo sum' mean in English? Brief.",
    # --- Edge cases (extra-short / extra-long expected output) ---
    "What is the next number in the sequence 2, 4, 8, 16, ...? Just the number.",
    "Give me a one-word adjective.",
    "Reply with only an exclamation mark.",
    "Spell 'cat' backwards. Just the answer.",
    "What letter comes after Z in the English alphabet? Brief.",
    # --- Coding-style follow-ups (reasoning model's bread and butter) ---
    "What's the time complexity of bubble sort? Just the big-O.",
    "What's the time complexity of binary search? Just the big-O.",
    "Is JSON a programming language? One word.",
    "What's the difference between TCP and UDP in one sentence?",
    "What HTTP status code means 'Not Found'? Just the number.",
    # --- Numerical reasoning at the edge of capacity ---
    "Estimate how many seconds are in a year. Just the rough number.",
    "Estimate how many words are in a typical novel. Rough number.",
    "Estimate the population of Tokyo. Rough number.",
    "Estimate the speed of light in km/s. Rough number.",
    "Estimate how many books are in the Library of Congress. Rough.",
    # --- Pattern recognition ---
    "What comes next: A, C, E, G, ...? Just the letter.",
    "What comes next: 1, 1, 2, 3, 5, ...? Just the number.",
    "What's the missing number: 2, 4, _, 8, 10? Just the number.",
    "What's the pattern: red, orange, yellow, ...? Brief.",
    "What comes next: Monday, Wednesday, Friday, ...? One word.",
    # --- Coding gotchas ---
    "In Python, what does `[] == False` evaluate to? Just the word.",
    "In JavaScript, what does `0 == '0'` evaluate to? Just the word.",
    "What does `1 << 4` equal in most languages? Just the number.",
    "What is the result of `7 / 2` in Python 3? Just the number.",
    "What is the result of `7 // 2` in Python? Just the number.",
]

# ----------------------------------------------------------------------------
# Tool-calling corpus (mode=tools)
#
# Each scenario is (user_prompt, list_of_tools).
# Tools follow OpenAI function-calling schema. The probe will scan both
# `content` AND every emitted `tool_calls[*].function.{name, arguments}` for
# raw <think>/<reasoning> leaks.
# ----------------------------------------------------------------------------

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                "unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
}

TOOL_SEARCH_WEB = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the public web and return top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
}

TOOL_SEND_EMAIL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email message.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

TOOL_RUN_SQL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Execute a read-only SQL query against the analytics warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT statement"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            "required": ["query"],
        },
    },
}

TOOL_CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2*(3+4)'"},
            },
            "required": ["expression"],
        },
    },
}

TOOL_CREATE_FILE = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a new file with given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}

TOOL_LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
}

TOOL_GET_STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'"},
            },
            "required": ["ticker"],
        },
    },
}

TOOL_TRANSLATE = {
    "type": "function",
    "function": {
        "name": "translate",
        "description": "Translate text between languages.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
}

TOOL_CREATE_TICKET = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": "Create a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "description"],
        },
    },
}

# All-tools toolbox (the model has to pick the right one).
ALL_TOOLS = [
    TOOL_GET_WEATHER, TOOL_SEARCH_WEB, TOOL_SEND_EMAIL, TOOL_RUN_SQL,
    TOOL_CALCULATOR, TOOL_CREATE_FILE, TOOL_LIST_FILES, TOOL_GET_STOCK,
    TOOL_TRANSLATE, TOOL_CREATE_TICKET,
]

TOOL_SCENARIOS = [
    # Single-tool, simple
    ("What's the weather in Tokyo right now? Use the tool.", [TOOL_GET_WEATHER]),
    ("How hot is it in Phoenix today? Use the tool. Use Fahrenheit.", [TOOL_GET_WEATHER]),
    ("Weather in Reykjavik in Celsius?", [TOOL_GET_WEATHER]),
    # Web search
    ("Search the web for 'best practices for kubernetes ingress'.", [TOOL_SEARCH_WEB]),
    ("Look up the latest news on the James Webb telescope. Give me 5 results.", [TOOL_SEARCH_WEB]),
    # Email
    ("Email alice@example.com with subject 'lunch?' and body 'free tomorrow at 1pm?'.", [TOOL_SEND_EMAIL]),
    ("Send a reminder to bob@example.com about the 3pm meeting. CC carol@example.com.", [TOOL_SEND_EMAIL]),
    # SQL
    ("Run a SQL query to count active users in the past 7 days.", [TOOL_RUN_SQL]),
    ("Run SQL: find the top 10 products by revenue this month.", [TOOL_RUN_SQL]),
    # Calculator
    ("Calculate 17 * 23 + 89.", [TOOL_CALCULATOR]),
    ("What is (5! - 3!) / 2? Use the calculator.", [TOOL_CALCULATOR]),
    # File ops
    ("Create a file at /tmp/hello.txt containing 'Hello, world!'.", [TOOL_CREATE_FILE]),
    ("List the files in /var/log recursively.", [TOOL_LIST_FILES]),
    # Stock
    ("What's the current price of NVDA?", [TOOL_GET_STOCK]),
    ("Get me AAPL's stock price.", [TOOL_GET_STOCK]),
    # Translate
    ("Translate 'hello world' to Japanese.", [TOOL_TRANSLATE]),
    ("Translate the sentence 'I love coffee in the morning' from English to French.", [TOOL_TRANSLATE]),
    # Ticket
    ("Create a high-priority support ticket: title 'Login broken', description 'Users cannot log in after the 10am deploy.'", [TOOL_CREATE_TICKET]),
    ("File a critical ticket about database replication lag, tagged 'db' and 'infra'.", [TOOL_CREATE_TICKET]),
    # Multi-tool: model must choose
    ("What's the weather in Paris?", ALL_TOOLS),
    ("Search for 'rust async runtime comparison'.", ALL_TOOLS),
    ("What is 2^32?", ALL_TOOLS),
    ("Email john@example.com saying I'll be late.", ALL_TOOLS),
    ("Get the stock price of GOOGL.", ALL_TOOLS),
    ("Translate 'thank you' into Korean.", ALL_TOOLS),
    ("Create a low-priority ticket titled 'typo on docs page'.", ALL_TOOLS),
    # Multi-step / harder reasoning before tool call
    ("If a customer asks for the weather and a stock price, call both tools. Customer asks: 'how's the weather in Seattle, and what's MSFT trading at?'", ALL_TOOLS),
    ("First check the weather in London, then if it's above 20c suggest a picnic. Just call the weather tool.", ALL_TOOLS),
    # Edge cases (model might decide no tool is needed)
    ("What is 2+2?", ALL_TOOLS),  # trivial, may answer without tool
    ("Hello!", ALL_TOOLS),  # greeting, no tool needed
]

# ----------------------------------------------------------------------------
# Leak detection
# ----------------------------------------------------------------------------

LEAK_PATTERNS = [
    re.compile(r"<\s*think\b", re.IGNORECASE),
    re.compile(r"</\s*think\s*>", re.IGNORECASE),
    re.compile(r"<\s*reasoning\b", re.IGNORECASE),
    re.compile(r"</\s*reasoning\s*>", re.IGNORECASE),
]


def looks_leaky(text: str) -> list[str]:
    """Return the list of leak-pattern matches found in `text` (empty if clean)."""
    if not text:
        return []
    hits = []
    for pat in LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def scan_message_for_leaks(msg: dict) -> dict:
    """Scan all user-visible fields of an OpenAI chat completion message for leaks.

    Returns a dict with per-field hit lists.
    """
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    out = {
        "content": looks_leaky(content),
        "tool_call_names": [],
        "tool_call_arguments": [],
    }
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments") or ""
        out["tool_call_names"].append(looks_leaky(name))
        out["tool_call_arguments"].append(looks_leaky(args))
    return out


def any_leak(scan: dict) -> bool:
    if scan["content"]:
        return True
    for hits in scan["tool_call_names"]:
        if hits:
            return True
    for hits in scan["tool_call_arguments"]:
        if hits:
            return True
    return False


def flat_patterns(scan: dict) -> list[str]:
    out = list(scan["content"])
    for hits in scan["tool_call_names"]:
        out.extend(hits)
    for hits in scan["tool_call_arguments"]:
        out.extend(hits)
    return out


# ----------------------------------------------------------------------------
# HTTP fire path
# ----------------------------------------------------------------------------

def _post(endpoint: str, payload: dict, timeout: float = 300.0) -> tuple[dict | None, str | None, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return None, f"{type(e).__name__}: {e}", round(time.time() - t0, 3)
    dt = round(time.time() - t0, 3)
    try:
        d = json.loads(body)
    except json.JSONDecodeError as e:
        return None, f"JSON decode: {e} | raw[:300]={body[:300]!r}", dt
    return d, None, dt


def fire_chat(endpoint: str, model: str, prompt: str, max_tokens: int, idx: int, seed: int, temperature: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
    }
    d, err, dt = _post(endpoint, payload)
    if err:
        return {"idx": idx, "mode": "chat", "prompt": prompt, "error": err, "latency_s": dt}
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    finish = (d.get("choices") or [{}])[0].get("finish_reason")
    scan = scan_message_for_leaks(msg)
    return {
        "idx": idx,
        "mode": "chat",
        "prompt": prompt,
        "finish_reason": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "content": content,
        "reasoning_head": reasoning[:200],
        "scan": scan,
        "leaks": flat_patterns(scan),
        "leaky": any_leak(scan),
        "latency_s": dt,
        "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
    }


def fire_tools(endpoint: str, model: str, prompt: str, tools: list[dict], max_tokens: int, idx: int, seed: int, temperature: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
    }
    d, err, dt = _post(endpoint, payload)
    if err:
        return {"idx": idx, "mode": "tools", "prompt": prompt, "error": err, "latency_s": dt}
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    finish = (d.get("choices") or [{}])[0].get("finish_reason")
    tool_calls = msg.get("tool_calls") or []
    scan = scan_message_for_leaks(msg)
    # Compact tool-call summary for log output
    tc_summary = [
        {"name": (tc.get("function") or {}).get("name"), "args": (tc.get("function") or {}).get("arguments")}
        for tc in tool_calls
    ]
    return {
        "idx": idx,
        "mode": "tools",
        "prompt": prompt,
        "finish_reason": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "n_tool_calls": len(tool_calls),
        "tool_calls": tc_summary,
        "content": content,
        "reasoning_head": reasoning[:200],
        "scan": scan,
        "leaks": flat_patterns(scan),
        "leaky": any_leak(scan),
        "latency_s": dt,
        "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_trials(mode: str, n: int, repeat: int) -> list[tuple[int, str, list[dict] | None, int]]:
    """Return a list of (global_idx, prompt, tools_or_None, seed) tuples."""
    if mode == "chat":
        base = CHAT_PROMPTS[:n]
        trials = []
        for r in range(repeat):
            for i, p in enumerate(base):
                gidx = r * len(base) + i
                trials.append((gidx, p, None, 42 + gidx))
        return trials
    elif mode == "tools":
        base = TOOL_SCENARIOS[:n]
        trials = []
        for r in range(repeat):
            for i, (prompt, tools) in enumerate(base):
                gidx = r * len(base) + i
                trials.append((gidx, prompt, tools, 42 + gidx))
        return trials
    else:
        raise SystemExit(f"unknown mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chat", "tools"], default="chat",
                    help="Probe mode: 'chat' = plain /v1/chat/completions, 'tools' = tool-calling traffic")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="Config label, e.g. fp8-mtp3-tools")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of base scenarios per pass (default: full corpus)")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat the corpus R times with unique seeds (trial total = N * R)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.mode == "chat":
        max_n = len(CHAT_PROMPTS)
    else:
        max_n = len(TOOL_SCENARIOS)
    n = args.n if args.n is not None else max_n
    if n > max_n:
        print(f"[leak-probe] WARNING: --n {n} > corpus size {max_n}, clamping", flush=True)
        n = max_n

    os.makedirs(args.out_dir, exist_ok=True)
    trials = build_trials(args.mode, n, args.repeat)
    t0 = time.time()
    results: list[dict] = []
    print(
        f"[leak-probe] mode={args.mode} label={args.label} endpoint={args.endpoint} model={args.model}",
        flush=True,
    )
    print(
        f"[leak-probe] n_scenarios={n} repeat={args.repeat} n_trials={len(trials)} "
        f"concurrency={args.concurrency} max_tokens={args.max_tokens} temperature={args.temperature}",
        flush=True,
    )

    def submit(ex: cf.ThreadPoolExecutor, gidx: int, prompt: str, tools, seed: int):
        if args.mode == "chat":
            return ex.submit(fire_chat, args.endpoint, args.model, prompt, args.max_tokens, gidx, seed, args.temperature)
        else:
            return ex.submit(fire_tools, args.endpoint, args.model, prompt, tools, args.max_tokens, gidx, seed, args.temperature)

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {submit(ex, gidx, prompt, tools, seed): gidx for (gidx, prompt, tools, seed) in trials}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "LEAK" if r.get("leaky") else ("ERR " if r.get("error") else "ok  ")
            if args.mode == "tools":
                tc = r.get("tool_calls") or []
                preview = ",".join(f"{t['name']}({(t['args'] or '')[:30]})" for t in tc) or (r.get("content") or r.get("error") or "")[:60]
            else:
                preview = (r.get("content") or r.get("error") or "")[:60]
            preview = preview.replace("\n", " ")
            print(
                f"[leak-probe] {tag} #{r['idx']:>3} ({r.get('latency_s','?')}s) {preview!r}",
                flush=True,
            )

    results.sort(key=lambda r: r["idx"])
    wall = round(time.time() - t0, 1)
    leaks = [r for r in results if r.get("leaky")]
    errors = [r for r in results if r.get("error")]

    # Collect per-surface leak counts (only meaningful in tools mode but harmless in chat)
    leak_in_content = sum(1 for r in leaks if r.get("scan", {}).get("content"))
    leak_in_tc_names = sum(
        1 for r in leaks if any(h for h in r.get("scan", {}).get("tool_call_names", []))
    )
    leak_in_tc_args = sum(
        1 for r in leaks if any(h for h in r.get("scan", {}).get("tool_call_arguments", []))
    )

    summary = {
        "label": args.label,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "model": args.model,
        "n_scenarios": n,
        "repeat": args.repeat,
        "n_trials": len(trials),
        "n_concurrent": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "wall_s": wall,
        "n_leaks_total": len(leaks),
        "n_leaks_in_content": leak_in_content,
        "n_leaks_in_tool_call_names": leak_in_tc_names,
        "n_leaks_in_tool_call_arguments": leak_in_tc_args,
        "leak_rate": round(len(leaks) / max(1, len(trials)), 4),
        "n_errors": len(errors),
        "leaky_indices": [r["idx"] for r in leaks],
        "error_indices": [r["idx"] for r in errors],
        "patterns_seen": sorted({p for r in leaks for p in r.get("leaks", [])}),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "responses.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "leaks.jsonl"), "w") as f:
        for r in leaks:
            f.write(json.dumps(r) + "\n")

    print("", flush=True)
    print(f"[leak-probe] === {args.label} ({args.mode}) summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"[leak-probe] artifacts: {args.out_dir}/summary.json, responses.jsonl, leaks.jsonl",
        flush=True,
    )
    return 0 if summary["n_leaks_total"] == 0 and summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

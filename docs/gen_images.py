#!/usr/bin/env python3
"""Generate hero + section images for the v3 study via Gemini 3.1 nano banana.

Matches the aesthetic of the hub's docs/gen_images.py (deep navy + electric cyan
/ amber accents, cinematic technical-poster style, no readable text). Three
illustrations:

  1. study_hero.png      — Tier-3 v3 stress-validation overview
  2. leak_probe.png      — dual-mode leak probe (chat + tool-calling) clean run
  3. production_incident.png — MTP/use_local_argmax_reduction post-mortem
"""
import os
import pathlib
import sys

from google import genai

OUT = pathlib.Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
)
MODEL = "gemini-3.1-flash-image-preview"

PROMPTS = {
    "study_hero": (
        "A wide cinematic 16:9 hero banner for a single deep-dive ML inference "
        "benchmark study titled (visually only, no text) 'v3 stress validation'. "
        "Centered foreground: a glowing 4-tile result matrix arranged in a 2x2 grid, "
        "each tile a different accent color (cyan, amber, magenta, lime) and stylized "
        "with abstract chart silhouettes inside, representing four model configurations "
        "(BF16+DFlash, FP8+DFlash, FP8+MTP=3, FP8+MTP=5). Behind the matrix, dual "
        "NVIDIA Blackwell data-center GPUs are visible in soft bokeh, with subtle "
        "tensor-parallel link traces flowing between them. Above the matrix a thin "
        "horizontal timeline glows showing a long sustained run. Deep navy and "
        "charcoal background, electric cyan and amber accents. Clean modern "
        "technical-poster aesthetic, precise infographic feel with cinematic 3D render. "
        "No readable text, no logos, no watermarks."
    ),
    "leak_probe": (
        "A wide 16:9 illustration of a permanent leak-detection probe running against "
        "a deployed language-model inference server. Left side: a stylized streaming "
        "pipe or river of tiny abstract token cards flowing horizontally, each card "
        "glowing pale cyan. Inside the stream, a translucent filter or sieve catches "
        "any 'thought' tokens (subtle violet shapes) and routes them downward into a "
        "separate reasoning channel, while clean content tokens continue rightward. "
        "Right side: two parallel test lanes labeled visually (no text) by distinct "
        "icon glyphs — one lane shows a plain chat bubble icon, the other shows a "
        "wrench-and-gear (tool/function-calling) icon. Both lanes terminate in glowing "
        "green checkmark seals indicating zero leaks. Subtle counter-style numerical "
        "indicators in the background (no readable text, just glyph-like marks). "
        "Deep navy backdrop, electric cyan + amber + lime accents, precise modern "
        "infographic-meets-cinematic style. No readable text, no logos, no watermarks."
    ),
    "production_incident": (
        "A 16:9 abstract forensic illustration of a production rollout incident, with "
        "NO TEXT of any kind anywhere in the image. Centered: a glowing 3D-rendered "
        "crystalline manifest tablet hovering in space, made of translucent layered "
        "glass plates, with one specific horizontal slot inside it glowing in bright "
        "red as the root cause flag. AVOID any letters, characters, numerals, code, "
        "glyphs that look like text, or pseudo-text squiggles. Use only pure geometric "
        "shapes: thin horizontal bars, dots, and tick marks to suggest data lines. "
        "Behind the tablet, a split diagnostic scene: on the left, a chaotic spiral of "
        "small warning-triangle icons in amber and red rendering a crash-loop; on the "
        "right, a clean steady horizontal line of green pulse beacons rendering a "
        "healthy server. Between the two halves, a single bold glowing arc transitions "
        "amber-to-green as the fix. In the deep background, two dual NVIDIA Blackwell-"
        "class GPUs sit on a rack in soft bokeh. Deep navy and charcoal palette, "
        "electric cyan structural light, amber+red on the failure side, transitioning "
        "to bright lime green on the success side. Clean cinematic technical-poster "
        "aesthetic, precise infographic detail. "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO NUMBERS, NO LOGOS, NO WATERMARKS, NO LABELS."
    ),
}

for name, prompt in PROMPTS.items():
    out_path = OUT / f"{name}.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
        continue
    print(f"[gen]  {name}: {prompt[:80]}...")
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        wrote = False
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                print(f"[ok]   {out_path} ({out_path.stat().st_size} bytes)")
                wrote = True
                break
        if not wrote:
            print(f"[warn] no image data for {name}")
    except Exception as e:
        print(f"[err]  {name}: {e}", file=sys.stderr)
        sys.exit(1)

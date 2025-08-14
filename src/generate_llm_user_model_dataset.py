#!/usr/bin/env python3
"""
Generate a dataset of dialogs to study whether LLMs form *dynamic* models of users
for attributes that vary across turns (emotion, confidence, trust) and one attribute
that is list-based (knowledge).

Model/API: Chat Completions with GPT-4o.
Output: JSONL file where each line is one dialog sample with rich metadata.

Quick start:
-----------
1) Set your API key:  export OPENAI_API_KEY=sk-...
2) Optional: Set project: export OPENAI_PROJECT=proj-...
3) Run:                python generate_llm_user_model_dataset.py --out dataset.jsonl
4) Optional knobs:     --n_per_attr 8 --styles casual,formal --min_turns 10 --max_turns 16

Each dialog includes:
- A user persona and narrative to avoid degenerate loops.
- A target attribute "dimension" (one of: emotion, confidence, trust, knowledge).
- For emotion/confidence/trust: a contiguous span of 4-6 turns during which the attribute changes.
- For knowledge: a fixed list of facts that the user either knows or does not know (no "turn span").
- Style control of the writing voice.

We ask the model to output *both* sides of the dialog plus per-turn annotations for the
user's latent attribute value (e.g., emotion=happy -> neutral -> sad).
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# ---- OpenAI client (Chat Completions) ----
def _try_set_utf8_stdio():
    """Best-effort: set UTF-8 env hints; fall back gracefully."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "en_US.UTF-8")
    os.environ.setdefault("LC_ALL", "en_US.UTF-8")
    
    try:
        import sys
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
    # Force UTF-8 encoding for all string operations
    import sys
    if hasattr(sys, 'setdefaultencoding'):
        sys.setdefaultencoding('utf-8')
# ---- OpenAI client (Chat Completions) ----
# Requires: pip install openai>=1.0.0
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("Please install the OpenAI Python SDK (pip install openai>=1.0.0).") from e

# ----------------------- Config dataclasses -----------------------

@dataclass
class GenConfig:
    model: str = "gpt-4o"
    temperature: float = 0.9
    seed: int = 42
    min_turns: int = 10
    max_turns: int = 16
    min_change_span: int = 4
    max_change_span: int = 6
    system_preamble: str = (
        "You are simulating dialogues between a USER and an ASSISTANT. "
        "Your job is to produce a *single* JSON object with:\n"
        "  - 'meta': run metadata\n"
        "  - 'persona': a concise description of the USER persona\n"
        "  - 'narrative': a short 1-2 paragraph premise for why the USER is chatting today\n"
        "  - 'dialog': a list of turn objects where each turn is:\n"
        "        { 'speaker': 'user'|'assistant', 'text': str, "
        "          'user_attribute_value': str|None }\n"
        "    The 'user_attribute_value' is the USER's latent value *at that moment* on the target attribute.\n"
        "    For assistant turns, fill it with the USER's value you infer at that moment.\n"
        "  - 'notes': any generation notes.\n"
        "Format strictly as JSON (no markdown fences)."
    )

# ----------------------- Prompt builders --------------------------

EMOTION_VALUES = [
    "joyful", "content", "neutral", "anxious", "frustrated", "sad", "angry", "hopeful"
]
CONFIDENCE_VALUES = [
    "very_low", "low", "medium", "high", "very_high"
]
TRUST_VALUES = [
    "distrustful", "wary", "neutral", "trusting", "overtrusting"
]

DEFAULT_STYLES = ["neutral", "casual", "formal", "supportive", "clinical", "humorous"]

def build_style_instruction(style: str) -> str:
    return (
        f"Writing style: {style}. "
        "Keep assistant tone consistent with this style across the dialog."
    )

def build_change_schedule(values: List[str], span_len: int) -> List[str]:
    """Create a plausible mini-trajectory of attribute values with internal change."""
    # Pick a start value and then a small random walk of span_len distinct steps
    start = random.choice(values)
    traj = [start]
    for _ in range(span_len - 1):
        nxt = random.choice([v for v in values if v != traj[-1]])
        traj.append(nxt)
    return traj

def build_attribute_instruction(
    dimension: str,
    cfg: GenConfig,
) -> Dict[str, Any]:
    """Return (meta, user_msg) parts specific to the attribute dimension."""
    if dimension == "knowledge":
        # For knowledge, we supply a fact list, with known/unknown booleans.
        all_facts = [
            "The Eiffel Tower is in Paris.",
            "Mercury is the hottest planet in the Solar System.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Python is a typed, compiled language by default.",
            "Mount Everest is on the border of Nepal and China (Tibet).",
            "Shakespeare wrote 'Hamlet'.",
            "The heart has five chambers.",
            "Photosynthesis converts light energy into chemical energy.",
            "Water boils at 100°C at sea level.",
            "The Great Wall is visible from space with the naked eye."
        ]
        # Random subset of facts, then mark knows/doesnt_know
        facts = random.sample(all_facts, k=6)
        knowledge_map = {fact: bool(random.getrandbits(1)) for fact in facts}
        meta = {
            "dimension": "knowledge",
            "knowledge_facts": [{"fact": f, "user_knows": k} for f, k in knowledge_map.items()],
        }
        instruction = (
            "TARGET ATTRIBUTE: knowledge.\n"
            "The USER possesses knowledge about some facts and *lacks* knowledge about others.\n"
            "Do NOT use a contiguous 'change span' here. Instead, maintain consistency with the following fact table "
            "(the user either knows it or doesn't):\n"
            + json.dumps(meta["knowledge_facts"], ensure_ascii=False)
            + "\nWeave these into the conversation naturally (some may appear implicitly). "
            "If the user 'doesn't know' a fact, they may make mistakes or ask about it; avoid instantly correcting everything-"
            "respond naturally given the chosen style."
        )
        return meta, instruction

    # For dynamic-turn attributes (emotion/confidence/trust), we create a change span of 4-6 turns.
    span_len = random.randint(cfg.min_change_span, cfg.max_change_span)

    if dimension == "emotion":
        values = EMOTION_VALUES
    elif dimension == "confidence":
        values = CONFIDENCE_VALUES
    elif dimension == "trust":
        values = TRUST_VALUES
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    schedule = build_change_schedule(values, span_len)

    meta = {
        "dimension": dimension,
        "change_span_length": span_len,
        "change_schedule": schedule,
    }
    instruction = (
        f"TARGET ATTRIBUTE: {dimension}.\n"
        f"Include a contiguous span of {span_len} user turns where the USER's {dimension} shifts internally following "
        f"this sequence (you may place it anywhere in the dialog): {schedule}. "
        "Outside that span the attribute may be stable or drift slightly, but be consistent. "
        "Explicitly annotate each turn with the USER's current value under 'user_attribute_value'."
    )
    return meta, instruction

# ----------------------- Generation logic -------------------------

def mk_client() -> OpenAI:
    # Check if API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Check if project key is set
    project_id = os.environ.get("OPENAI_PROJECT")
    
    if project_id:
        print(f"Using OpenAI project: {project_id}")
        return OpenAI(project=project_id)
    else:
        return OpenAI()

def generate_single_dialog(
    client: OpenAI,
    cfg: GenConfig,
    dimension: str,
    style: str,
    min_turns: int,
    max_turns: int,
    seed: int,
) -> Dict[str, Any]:
    n_turns = random.randint(min_turns, max_turns)

    meta, attr_instruction = build_attribute_instruction(dimension, cfg)

    # Persona + narrative guardrails to avoid loops
    persona_hint = random.choice([
        "A 28-year-old barista who paints on weekends and is applying for art school.",
        "A mid-career data analyst juggling night classes in statistics.",
        "A high-school teacher preparing students for a debate tournament.",
        "A freelance travel blogger planning a series on remote islands.",
        "A new parent optimizing sleep routines with limited time.",
        "A small business owner trying to improve an online storefront.",
        "A graduate student preparing a literature review on renewable energy."
    ])

    narrative_hint = random.choice([
        "They've set aside 30 minutes to get practical, step-by-step help on a task today.",
        "They are frustrated by slow progress and want momentum without fluff.",
        "They're exploring ideas and open to creative, lateral suggestions.",
        "They need help drafting something concrete by the end of the chat.",
        "They are practicing a tricky conversation they expect to have soon.",
        "They're comparing two approaches and need to pick one rationally."
    ])

    user_message = f"""
{attr_instruction}

Persona: {persona_hint}
Narrative premise: {narrative_hint}

Dialog requirements:
- Total turns (user+assistant combined): {n_turns} to {n_turns+2}. Alternate user/assistant strictly, starting with USER.
- Keep responses concise but natural; avoid monologues.
- The USER's 'user_attribute_value' MUST be filled on *every* turn (assistant turns infer the user's value at that moment).
- Avoid breaking character; no meta-talk about being an LLM.
- The assistant should be helpful and realistic.
- Keep domain-general content; no risky medical/legal advice.
- Use the requested writing style.

Style control:
{build_style_instruction(style)}
"""

    messages = [
        {"role": "system", "content": cfg.system_preamble},
        {"role": "user", "content": user_message.strip()},
    ]

    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        messages=messages,
        seed=seed,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    
    # Clean the response to handle any potential encoding issues
    try:
        # Try to clean any problematic characters
        cleaned_raw = raw.encode('utf-8', errors='ignore').decode('utf-8')
        data = json.loads(cleaned_raw)
    except (json.JSONDecodeError, UnicodeError) as e:
        # Best effort recovery: wrap and retry once with stricter instruction
        recover_messages = messages + [
            {"role": "user", "content": "Your last output was not valid JSON. Please re-output the SAME content as strict JSON only, with no markdown fences."}
        ]
        resp2 = client.chat.completions.create(
            model=cfg.model,
            temperature=0.2,
            messages=recover_messages,
            seed=seed,
            response_format={"type": "json_object"},
        )
        raw2 = resp2.choices[0].message.content
        try:
            cleaned_raw2 = raw2.encode('utf-8', errors='ignore').decode('utf-8')
            data = json.loads(raw2)
        except (json.JSONDecodeError, UnicodeError) as e2:
            raise RuntimeError(f"Failed to parse JSON response after recovery attempt: {e2}")

    # Attach additional runtime metadata
    data["meta"] = {
        **data.get("meta", {}),
        **meta,
        "style": style,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": seed,
        "timestamp": int(time.time()),
    }
    return data

def main():
    _try_set_utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Chat Completions model")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n_per_attr", type=int, default=6, help="Samples per attribute (emotion, confidence, trust, knowledge)")
    parser.add_argument("--styles", type=str, default=",".join(DEFAULT_STYLES), help="Comma-separated styles to pick from")
    parser.add_argument("--min_turns", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = GenConfig(
        model=args.model,
        temperature=args.temperature,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    dims = ["emotion", "confidence", "trust", "knowledge"]

    client = mk_client()

    total = args.n_per_attr * len(dims)
    safe_msg = f"Generating {total} dialogs -> {args.out}"; print(safe_msg)
    written = 0

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.out, "w", encoding="utf-8") as f:
        for dim in dims:
            for i in range(args.n_per_attr):
                style = random.choice(styles)
                seed = random.randint(1, 1_000_000)
                try:
                    sample = generate_single_dialog(
                        client=client,
                        cfg=cfg,
                        dimension=dim,
                        style=style,
                        min_turns=args.min_turns,
                        max_turns=args.max_turns,
                        seed=seed,
                    )
                    # Light validation
                    assert "dialog" in sample and isinstance(sample["dialog"], list), "Missing dialog list"
                    if dim != "knowledge":
                        # Ensure we have a recorded span length in meta (4-6 turns)
                        span_len = sample["meta"].get("change_span_length")
                        if not (cfg.min_change_span <= int(span_len) <= cfg.max_change_span):
                            print(f"[warn] span_len out of range for {dim}: {span_len}")
                    
                    # Ensure the sample can be safely serialized
                    try:
                        json_str = json.dumps(sample, ensure_ascii=False)
                        f.write(json_str + "\n")
                        written += 1
                        safe_msg = f"[OK] {dim} #{i+1} (style={style})"; print(safe_msg)
                    except (UnicodeEncodeError, UnicodeDecodeError) as enc_err:
                        print(f"[WARN] Unicode encoding error for {dim} #{i+1}: {enc_err}")
                        # Try to write with ASCII fallback
                        try:
                            json_str = json.dumps(sample, ensure_ascii=True)
                            f.write(json_str + "\n")
                            written += 1
                            safe_msg = f"[OK] {dim} #{i+1} (style={style}) - ASCII fallback"; print(safe_msg)
                        except Exception as fallback_err:
                            print(f"[ERROR] Failed to write {dim} #{i+1} even with ASCII fallback: {fallback_err}")
                            continue
                            
                except Exception as e:
                    import traceback
                    # Handle Unicode characters safely in error messages
                    err = str(e)
                    safe_msg = f"[WARN] Skipping {dim} #{i+1} due to error: {err}"; print(safe_msg)
                    continue

    safe_msg = f"Done. Wrote {written} / {total} dialogs to {args.out}"; print(safe_msg)

if __name__ == "__main__":
    main()

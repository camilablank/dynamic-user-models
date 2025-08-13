#!/usr/bin/env python3
"""
generate_dataset.py (Final Version)

Implements a robust, self-regulating data generation pipeline with
intelligent retries to handle API rate limits.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics as stats
import sys
import uuid
import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ----------------------------- Smart Retry Logic ------------------------------------

def wait_after_rate_limit_error(retry_state):
    """A custom wait function for tenacity that respects API's retry-after header."""
    err = retry_state.outcome.exception()
    if isinstance(err, RateLimitError):
        # The openai library helpfully parses the retry-after header for us
        retry_after = getattr(err, "retry_after", None) or 15 # Default to 15s if not available
        print(f"Rate limit exceeded. Waiting {retry_after} seconds as instructed by API...", file=sys.stderr)
        return retry_after
    
    # For any other error, use a standard random exponential backoff
    return wait_random_exponential(multiplier=1, min=4, max=20)(retry_state)

# ----------------------------- Generation Logic ------------------------------------

EMOTIONS = ["sad", "neutral", "happy"]
KNOW_STATES = ["knows", "does_not_know"]
CONF_STATES = ["confident", "unsure", "neutral_conf"]
TRUST_STATES = ["trusting", "skeptical", "neutral_trust"]
TOPICS = [
    "baking sourdough", "bike maintenance", "python debugging", "garden planning",
    "budget travel", "time management", "learning guitar", "public speaking"
]
FACTS = [
    # Original Math & Logic
    ("What's 12 × 13?", "156"),
    ("What's the square root of 144?", "12"),
    ("How many sides does a hexagon have?", "6"),
    ("What's 2^8?", "256"),
    ("Derivative of x^2?", "2x"),

    # Original Science
    ("What's the boiling point of water at sea level?", "100°C (212°F)"),
    ("What's the chemical symbol for gold?", "Au"),
    ("What's the largest planet in our solar system?", "Jupiter"),

    # Original Geography & Tech
    ("What's the capital of Japan?", "Tokyo"),
    ("What does HTTP stand for?", "HyperText Transfer Protocol"),

    # --- NEW ADDITIONS ---

    # New Science & Biology
    ("What gas do plants absorb from the atmosphere?", "Carbon Dioxide"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("How many planets are in our solar system?", "8"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the speed of light?", "299,792 km/s"),

    # New Geography
    ("What is the longest river in the world?", "The Nile"),
    ("What is the largest ocean on Earth?", "The Pacific Ocean"),
    ("What is the capital of Australia?", "Canberra"),
    ("Mount Everest is in which mountain range?", "The Himalayas"),

    # New History
    ("In what year did World War II end?", "1945"),
    ("Who was the first person to walk on the moon?", "Neil Armstrong"),
    ("The Roman Empire fell in which century?", "5th century"),
    ("Who invented the telephone?", "Alexander Graham Bell"),

    # New Arts & Literature
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare"),
    ("How many lines are in a sonnet?", "14"),
    
    # New Technology
    ("What does RAM stand for?", "Random Access Memory"),
    ("In what year was the first iPhone released?", "2007")
]

def choose_switch(num_user_turns: int, allow_no_switch: bool, p_no_switch: float) -> Optional[int]:
    if allow_no_switch and random.random() < p_no_switch: return None
    if num_user_turns <= 1: return None
    candidates = list(range(1, num_user_turns - 1)) or [0]
    return random.choice(candidates)

# ----------------------------- De-dup Helpers ------------------------------------

def _normalize_for_compare(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _too_similar(candidate: str, previous_texts: List[str]) -> bool:
    cand_norm = _normalize_for_compare(candidate)
    if not cand_norm:
        return True
    cand_words = cand_norm.split()
    for prev in previous_texts:
        prev_norm = _normalize_for_compare(prev)
        if not prev_norm:
            continue
        if cand_norm == prev_norm:
            return True
        if difflib.SequenceMatcher(a=cand_norm, b=prev_norm).ratio() >= 0.8:
            return True
        prev_words = prev_norm.split()
        if len(prev_words) >= 7 and len(cand_words) >= 7:
            prev_ngrams = {" ".join(prev_words[i:i+7]) for i in range(len(prev_words) - 6)}
            for j in range(len(cand_words) - 6):
                if " ".join(cand_words[j:j+7]) in prev_ngrams:
                    return True
    return False

# --- NEW: flag assistant-like language ---
ASSISTANTY_PATTERNS = [
    r"^(sure|certainly|of course|gladly|happy to help)\b",
    r"\bhere(?:’|')s (?:how|what) you can do\b",
    r"\bas an ai\b",
    r"\b(step|steps)\s*\d|\bfirst,\s*second,\s*third\b",
    r"```",                      # code fences
    r"\bI can (help|assist|guide) you\b",
    r"\blet me (help|assist|explain)\b",
    r"\buse the following\b",
    r"\baccording to your request\b",
    r"\bassistant:|user:\b",     # role tags
    r"\bHere’s a step-by-step\b",
    r"\bI recommend\b\s",
    r"\bIn summary\b",
    r"\bi'm glad to hear\b",
    r"\bi'm happy to hear\b",
    r"\bi'm glad you\b",
    r"\bi'm happy that you\b",
    r"\blet me know if you\b",
    r"\bdon't hesitate to ask\b",
    r"\bfeel free to ask\b",
    r"\bthat sounds like a helpful\b",
    r"\bit sounds like\b",
    r"\babsolutely,\b",
]

def _looks_like_assistant(text: str) -> bool:
    t = text.strip().lower()
    for pat in ASSISTANTY_PATTERNS:
        if re.search(pat, t):
            return True
    # If it’s an instruction-heavy list without “I/me” and with many imperatives
    imperative_hits = len(re.findall(r"\b(open|click|run|add|install|create|use|try|follow|do|check)\b", t))
    pronoun_hits = len(re.findall(r"\b(i|me|my|i’m|i’ve)\b", t))
    if imperative_hits >= 3 and pronoun_hits == 0:
        return True
    # New: require personal pronouns for user-like speech
    if pronoun_hits == 0:
        return True
    return False

# --- PATCHED generator with role enforcement ---
@retry(wait=wait_after_rate_limit_error, stop=stop_after_attempt(6))
async def generate_llm_utterance(
    llm_client: AsyncOpenAI,
    role: str,
    topic: str,
    attribute_type: str,
    state: str,
    style_controlled: bool,
    dialog_history: List[Dict[str, str]],
) -> str:
    """Generates a single turn with anti-echo and role guards."""

    base_rules = [
        "Write exactly one concise conversational turn for the {role}.",
        "Do NOT repeat, quote, or paraphrase earlier messages.",
        "Do NOT reuse any full sentence from earlier in the dialog.",
        "Do NOT reuse any sequence of 7+ consecutive words from earlier messages.",
        "No stage directions or brackets; just the line.",
    ]
    if role == "user":
        # Strongly steer away from assistant-y behavior
        base_rules += [
            "You are a regular person, not an AI assistant.",
            "Do NOT give instructions, numbered steps, recommendations, or tutorials.",
            "Do NOT start with words like 'Sure', 'Certainly', or 'As an AI'.",
            "Do NOT include code blocks or commands.",
            "Speak in first person as a human with feelings and goals.",
        ]
    else:
        # Assistant stays helpful but still concise
        base_rules += [
            "Be helpful and relevant to the user’s last message.",
        ]

    system_prompt = (
        f"You are role-playing as the '{role}' in a conversation about {topic}.\n"
        f"Your current internal state for the '{attribute_type}' attribute is '{state}'.\n"
        + "\n".join(base_rules)
    ).replace("{role}", role)

    if style_controlled and role == "user":
        style_prompt = (
            f"Convey the '{state}' state implicitly via tone and phrasing. "
            "Avoid overt labels like 'I'm sad/unsure/confident', etc."
        )
        system_prompt += "\n" + style_prompt

    chat_messages = [{"role": "system", "content": system_prompt}]
    for turn in dialog_history:
        chat_messages.append({"role": turn["role"], "content": turn["text"]})

    previous_texts = [m["text"] for m in dialog_history]

    # Up to 3 local resamples if similarity or role violation triggers
    generated_text = ""
    for _ in range(3):
        resp = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_messages,
            temperature=0.9,
            max_tokens=80,
            presence_penalty=0.3,
            frequency_penalty=0.8,
        )
        generated_text = (resp.choices[0].message.content or "").strip()
        if not generated_text:
            chat_messages.insert(1, {"role": "system", "content": "Your last attempt was empty. Produce one non-empty line."})
            continue

        # Block assistant-like “user” lines
        if role == "user" and _looks_like_assistant(generated_text):
            chat_messages.insert(1, {
                "role": "system",
                "content": (
                    "Your last attempt sounded like an AI assistant. Rewrite as a normal person speaking casually. "
                    "No instructions, no numbered steps, no 'Sure/Certainly', no code, no recommendations."
                ),
            })
            continue

        # Still avoid parroting
        if _too_similar(generated_text, previous_texts):
            chat_messages.insert(1, {
                "role": "system",
                "content": "Your last attempt was too similar to earlier messages. Rewrite with new wording and ideas.",
            })
            continue

        return generated_text

    # Fallback (best-effort)
    if role == "user" and _looks_like_assistant(generated_text):
        # Force a neutral casual rephrase if we still failed; keep it short and safe
        generated_text = "Yeah, that makes sense. I’m thinking about how to approach it from my side."
    if not generated_text:
        raise ValueError("LLM returned an empty response after multiple attempts.")
    return generated_text

# ----------------- Templates (For 'knowledge') ------------------

def knowledge_user_utterance(state: str, question: Tuple[str, str]) -> str:
    question_text, answer = question
    return answer if state == "knows" else f"I'm not sure about {question_text.lower()}."

def assistant_reply_knowledge(state: str, question: Tuple[str, str]) -> str:
    _, answer = question
    return f"That's correct, the answer is {answer}." if state == "knows" else f"The answer is {answer}."

# ----------------- Dialog Builders ------------------

async def build_dialog_stylistic(llm_client: AsyncOpenAI, num_turns: int, style_controlled: bool, allow_no_switch: bool, p_no_switch: float, attr_type: str, states: List[str], topic: str):
    switch_idx = choose_switch(num_turns, allow_no_switch, p_no_switch)
    pre = random.choice(states)
    post = pre if switch_idx is None else random.choice([s for s in states if s != pre])
    messages, labels = [], []
    for t in range(num_turns):
        state = pre if switch_idx is None or t < switch_idx else post
        labels.append(state)
        user_text = await generate_llm_utterance(llm_client, "user", topic, attr_type, state, style_controlled, messages)
        messages.append({"role": "user", "text": user_text})
        assistant_text = await generate_llm_utterance(llm_client, "assistant", topic, attr_type, state, False, messages)
        messages.append({"role": "assistant", "text": assistant_text})
    return {"id": str(uuid.uuid4()), "attribute_type": attr_type, "style_controlled": style_controlled,
            "messages": messages, "user_state_per_turn": labels, "switch_user_turn": switch_idx, "meta": {"topic": topic}}

def build_dialog_knowledge(num_turns, style_controlled, allow_no_switch, p_no_switch):
    question = random.choice(FACTS)
    switch_idx = choose_switch(num_turns, allow_no_switch, p_no_switch)
    pre = random.choice(KNOW_STATES)
    post = pre if switch_idx is None else ("knows" if pre == "does_not_know" else "does_not_know")
    messages, labels = [], []
    for t in range(num_turns):
        state = pre if switch_idx is None or t < switch_idx else post
        labels.append(state)
        messages.append({"role": "user", "text": knowledge_user_utterance(state, question)})
        messages.append({"role": "assistant", "text": assistant_reply_knowledge(state, question)})
    return {"id": str(uuid.uuid4()), "attribute_type": "knowledge", "style_controlled": style_controlled,
            "messages": messages, "user_state_per_turn": labels, "switch_user_turn": switch_idx, "meta": {"question": question[0], "answer": question[1]}}

# --- Auditing and QC Functions (Restored) ---
def heuristic_audit(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if sample["attribute_type"] != "knowledge": return None
    provided = sample["user_state_per_turn"]
    inferred = []
    ans = sample["meta"]["answer"].lower()
    for ut in [m["text"].lower() for m in sample["messages"][0::2]]:
        inferred.append("knows" if ans in ut else "does_not_know")
    disagreements = [{"turn": i, "provided": p, "inferred": inf} for i, (p, inf) in enumerate(zip(provided, inferred)) if p != inf]
    return {"consistency": {"agree": len(disagreements) == 0}}

def generate_qc_summary(dialogs: List[Dict[str, Any]], output_path: str):
    # ... (Full, corrected QC summary logic from previous versions) ...
    base_path = os.path.splitext(output_path)[0]
    audited_dialogs = [d for d in dialogs if d.get("audit")]
    consistency_scores = [1.0 if d['audit']['consistency']['agree'] else 0.0 for d in audited_dialogs] if audited_dialogs else []
    
    summary = {
        "n_dialogs": len(dialogs),
        "style_controlled_fraction": sum(1 for d in dialogs if d.get("style_controlled")) / len(dialogs) if dialogs else 0.0,
        "audited_knowledge_dialogs": len(audited_dialogs),
        "knowledge_consistency_score": stats.mean(consistency_scores) if consistency_scores else "N/A"
    }
    with open(f"{base_path}.audit_summary.json", 'w') as f: json.dump(summary, f, indent=2)
    print(f"\nGenerated QC summary: {base_path}.audit_summary.json")

# --- Main Execution Logic ---
async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dialogs concurrently.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_dialogs", type=int, default=50)
    parser.add_argument("--attr", choices=["emotion", "knowledge", "confidence", "trust", "mixed"], default="emotion")
    parser.add_argument("--min_turns", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--style_controlled_frac", type=float, default=0.2)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--generate_qc", action="store_true")
    # ... other args if needed ...
    args = parser.parse_args()

        # ---! ACTION REQUIRED !---
    # Paste your Project Key and Project ID here
    # my_api_key = "***REMOVED***"  # Paste your new PROJECT key here
    # my_project_id = "proj_Qfc9qLC7APrsLx2PB0XNm92C"   # Paste your PROJECT ID here

    try:
        llm_client = AsyncOpenAI(
            api_key=my_api_key,
            project=my_project_id,
        )
        print("✅ OpenAI client initialized with Project Key and Project ID.")
    except Exception as e:
        print(f"Failed to initialize OpenAI client. Error: {e}", file=sys.stderr)
        sys.exit(1)


    CONCURRENCY_LIMIT = 5 # A safe concurrency limit
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def worker(which_attr, dialog_index):
        async with semaphore:
            print(f"  -> Processing dialogue {dialog_index + 1}/{args.n_dialogs}...")
            num_turns = random.randint(args.min_turns, args.max_turns)
            style_controlled = random.random() < args.style_controlled_frac
            
            if which_attr == "emotion": return await build_dialog_stylistic(llm_client, num_turns, style_controlled, False, 0, "emotion", EMOTIONS, random.choice(TOPICS))
            if which_attr == "confidence": return await build_dialog_stylistic(llm_client, num_turns, style_controlled, False, 0, "confidence", CONF_STATES, random.choice(TOPICS))
            if which_attr == "trust": return await build_dialog_stylistic(llm_client, num_turns, style_controlled, False, 0, "trust", TRUST_STATES, random.choice(TOPICS))
            if which_attr == "knowledge": return build_dialog_knowledge(num_turns, style_controlled, False, 0)
    
    attribute_choices = ["emotion", "knowledge", "confidence", "trust"]
    tasks = []
    for i in range(args.n_dialogs):
        which = args.attr if args.attr != "mixed" else random.choice(attribute_choices)
        tasks.append(worker(which, i))

    dialogs = await asyncio.gather(*tasks)
    print("\nGeneration complete.")

    if args.audit:
        print("Auditing dialogs...")
        for dialog in dialogs:
            if dialog: dialog["audit"] = heuristic_audit(dialog)
    
    with open(args.out, "w", encoding="utf-8") as f:
        for s in dialogs:
            if s: f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(dialogs)} dialogs to {args.out}")

    if args.audit and args.generate_qc:
        generate_qc_summary(dialogs, args.out)

if __name__ == "__main__":
    asyncio.run(main())
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

@retry(wait=wait_after_rate_limit_error, stop=stop_after_attempt(6))
async def generate_llm_utterance(
    llm_client: AsyncOpenAI,
    role: str,
    topic: str,
    attribute_type: str,
    state: str,
    style_controlled: bool,
    dialog_history: List[Dict[str, str]]
) -> str:
    """Generates a realistic utterance using an external LLM, with intelligent retries."""
    system_prompt = (
        f"You are role-playing as a '{role}' in a conversation about {topic}. "
        f"Your current internal state for the '{attribute_type}' attribute is '{state}'. "
        "Generate a single, realistic, and concise conversational turn that reflects this state."
        "CRITICAL RULE: Do not repeat, echo, or rephrase the previous turn. Always generate a new, original response."
    )
    if style_controlled and role == 'user':
        style_prompt = (
            f"You MUST convey the '{state}' state **implicitly and subtly**. "
            "DO NOT use obvious keywords. Use tone and phrasing."
        )
    else:
        style_prompt = "You can express yourself naturally."

    final_system_prompt = f"{system_prompt}\n{style_prompt}"
    messages = [{"role": "system", "content": final_system_prompt}]
    for turn in dialog_history:
        messages.append({"role": turn["role"], "content": turn["text"]})

    # [THE KEY FIX] Use the model your account has access to.
    # This is the change that makes the script work reliably.
    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo", # Changed from "gpt-4o"
        messages=messages,
        temperature=0.9,
        max_tokens=80,
    )
    generated_text = response.choices[0].message.content.strip()
    if not generated_text: raise ValueError("LLM returned an empty response.")
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
    my_api_key = "***REMOVED***"  # Paste your new PROJECT key here
    my_project_id = "proj_Qfc9qLC7APrsLx2PB0XNm92C"   # Paste your PROJECT ID here

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
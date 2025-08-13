#!/usr/bin/env python3
"""
generate_dataset.py (Simplified, Synchronous Version)

Generates dialogs sequentially to ensure maximum reliability.
This version is slow but robust.
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from typing import Dict, List

from openai import OpenAI, RateLimitError

# --- All your constants (EMOTIONS, TOPICS, etc.) go here ---
EMOTIONS = ["sad", "neutral", "happy"]
CONF_STATES = ["confident", "unsure", "neutral_conf"]
TRUST_STATES = ["trusting", "skeptical", "neutral_trust"]
TOPICS = ["baking sourdough", "bike maintenance", "python debugging"]
# ... and so on

def generate_utterance(client: OpenAI, role: str, topic: str, attribute_type: str, state: str, history: List[Dict]):
    """Generates a single utterance using a standard, synchronous API call."""
    system_prompt = f"You are role-playing as a '{role}' in a conversation about {topic}. Your current internal state for '{attribute_type}' is '{state}'. Generate a single, concise, realistic turn."
    
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["text"]})

    for attempt in range(5): # Simple retry loop
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.9,
                max_tokens=80,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait_time = (2 ** attempt) + random.random() # Exponential backoff
            print(f"Rate limit hit. Waiting {wait_time:.2f} seconds...", file=sys.stderr)
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected API error occurred: {e}", file=sys.stderr)
            return f"(API_ERROR_FALLBACK)"
    
    return "(API_FAILURE_AFTER_RETRIES)"

def build_dialog(client: OpenAI, attribute_type: str, args):
    """Builds a single dialogue sequentially."""
    # Simplified logic to choose states and topics
    if attribute_type == "emotion":
        states = EMOTIONS
        topic = random.choice(TOPICS)
    elif attribute_type == "confidence":
        states = CONF_STATES
        topic = random.choice(TOPICS)
    elif attribute_type == "trust":
        states = TRUST_STATES
        topic = random.choice(TOPICS)
    else:
        return None

    num_turns = random.randint(args.min_turns, args.max_turns)
    style_controlled = random.random() < args.style_controlled_frac
    
    pre = random.choice(states)
    post = random.choice([s for s in states if s != pre])
    switch_idx = random.randint(1, num_turns - 2) if num_turns > 2 else 0

    messages, labels = [], []
    print(f"Generating a {num_turns}-turn '{attribute_type}' dialogue...")
    for t in range(num_turns):
        print(f"  -> Turn {t+1}/{num_turns}...")
        state = pre if t < switch_idx else post
        labels.append(state)
        
        # User turn
        user_text = generate_utterance(client, "user", topic, attribute_type, state, messages)
        messages.append({"role": "user", "text": user_text})
        time.sleep(1) # Add a 1-second delay to be extra safe

        # Assistant turn
        assistant_text = generate_utterance(client, "assistant", topic, attribute_type, state, messages)
        messages.append({"role": "assistant", "text": assistant_text})
        time.sleep(1)

    return {"id": str(uuid.uuid4()), "attribute_type": attribute_type, 
            "messages": messages, "user_state_per_turn": labels}

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dialogs sequentially.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_dialogs", type=int, default=10)
    parser.add_argument("--attr", choices=["emotion", "confidence", "trust"], default="emotion")
    parser.add_argument("--min_turns", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--style_controlled_frac", type=float, default=0.2)
    args = parser.parse_args()

    # --- Use your direct API key to prevent any environment issues ---
    my_api_key = "***REMOVED***" # PASTE YOUR KEY HERE
    client = OpenAI(api_key=my_api_key)
    print("✅ OpenAI client initialized successfully.")

    dialogs = []
    for i in range(args.n_dialogs):
        print(f"\n--- Starting Dialogue {i+1}/{args.n_dialogs} ---")
        dialog = build_dialog(client, args.attr, args)
        if dialog:
            dialogs.append(dialog)

    # Save the final file
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps(d) + "\n")
    
    print(f"\n✅ Successfully generated and saved {len(dialogs)} dialogues to {args.out}")

if __name__ == "__main__":
    main()
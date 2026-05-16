#!/usr/bin/env python3
"""
chat.py — Interactive CLI client for the Qwen3.6 LLM server on Kaggle

Usage:
    python chat.py --url https://xxxx.trycloudflare.com
    python chat.py --url https://xxxx.trycloudflare.com --thinking

Commands during conversation:
    /image path/to/file.jpg   → send an image with your next message
    /clear                    → clear conversation history (start fresh)
    /history                  → show the current conversation
    /exit  or  Ctrl+C         → quit
"""

import argparse
import base64
import json
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai package not found. Run: pip install openai")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    """Return a base64-encoded data URI for an image file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = p.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    mime = mime_map.get(suffix, "image/jpeg")
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}", mime


def build_user_message(text: str, image_path: str | None) -> dict:
    """
    Build the message dict for the user turn.
    If an image is attached, the content becomes a list (text + image).
    Otherwise it's a simple string.
    """
    if image_path:
        data_uri, _ = encode_image(image_path)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    return {"role": "user", "content": text}


def print_history(history: list):
    """Print the full conversation history in a readable format."""
    if not history:
        print("  (empty)")
        return
    for i, msg in enumerate(history):
        role = msg["role"].upper()
        if isinstance(msg["content"], str):
            preview = msg["content"][:120].replace("\n", " ")
        else:
            # multimodal content list
            text_parts = [c["text"] for c in msg["content"] if c.get("type") == "text"]
            has_image  = any(c.get("type") == "image_url" for c in msg["content"])
            preview = " ".join(text_parts)[:120]
            if has_image:
                preview += "  [+ image]"
        print(f"  [{i}] {role}: {preview}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat client for the Kaggle LLM server"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Your Cloudflare tunnel URL (e.g. https://xxxx.trycloudflare.com)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6-35b-a3b",
        help="Model alias to use (default: qwen3.6-35b-a3b)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (model reasons before answering)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt (default: 'You are a helpful assistant.')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response (default: 2048)",
    )
    args = parser.parse_args()

    # Normalize URL
    base_url = args.url.rstrip("/") + "/v1"

    # Connect to the server
    client = OpenAI(base_url=base_url, api_key="none")

    # Test connection before entering the chat loop
    print(f"\n🔌 Connecting to {base_url} ...")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"✅ Connected. Available model(s): {', '.join(available)}")
    except Exception as e:
        print(f"❌ Could not connect: {e}")
        print("   Is the server running? Did you get the URL from 'kaggle kernels output'?")
        sys.exit(1)

    # Build the initial conversation history with the system prompt
    history = [{"role": "system", "content": args.system}]

    mode_label = "🧠 thinking" if args.thinking else "⚡ direct"
    print(f"\n{'─'*55}")
    print(f"  Model   : {args.model}")
    print(f"  Mode    : {mode_label}")
    print(f"  System  : {args.system[:60]}{'...' if len(args.system)>60 else ''}")
    print(f"{'─'*55}")
    print("  Commands: /image <path>  /clear  /history  /exit")
    print(f"{'─'*55}\n")

    pending_image = None   # Path of an image queued for the next message

    while True:
        # ── Prompt ────────────────────────────────────────────────────────────
        try:
            if pending_image:
                prompt_label = f"[image: {Path(pending_image).name}] You: "
            else:
                prompt_label = "You: "
            user_input = input(prompt_label).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Bye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd   = parts[0].lower()

            if cmd == "/exit":
                print("👋 Bye!")
                break

            elif cmd == "/clear":
                history = [{"role": "system", "content": args.system}]
                pending_image = None
                print("🗑️  Conversation cleared.\n")

            elif cmd == "/history":
                print("\n📜 Conversation history:")
                print_history(history)
                print()

            elif cmd == "/image":
                if len(parts) < 2:
                    print("  Usage: /image path/to/file.jpg\n")
                else:
                    img_path = parts[1].strip()
                    try:
                        encode_image(img_path)   # validate
                        pending_image = img_path
                        print(f"  🖼️  Image queued: {img_path}")
                        print("  Now type your message and press Enter to send both.\n")
                    except FileNotFoundError as e:
                        print(f"  ❌ {e}\n")
            else:
                print(f"  ❓ Unknown command: {cmd}\n")

            continue

        # ── Build and send the message ─────────────────────────────────────────
        user_msg = build_user_message(user_input, pending_image)
        pending_image = None   # consumed

        history.append(user_msg)

        print("\nAssistant: ", end="", flush=True)

        try:
            # Stream the response token by token
            stream = client.chat.completions.create(
                model      = args.model,
                messages   = history,
                max_tokens = args.max_tokens,
                stream     = True,
                extra_body = {
                    # Thinking mode per request — no server restart needed
                    "chat_template_kwargs": {"enable_thinking": args.thinking},
                    # Adjust sampling params per Unsloth recommendations
                    # thinking=True  → temp 1.0, top_p 0.95 (more creative reasoning)
                    # thinking=False → temp 0.7, top_p 0.8  (focused direct answers)
                    "temperature": 1.0 if args.thinking else 0.7,
                    "top_p":       0.95 if args.thinking else 0.8,
                },
            )

            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_response += delta.content

            print("\n")   # newline after the streamed response

            # Add the assistant response to history for the next turn
            history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            # User pressed Ctrl+C mid-stream — don't add incomplete response
            print("\n\n⚠️  Response interrupted. History not updated for this turn.\n")
            history.pop()   # remove the user message we just added

        except Exception as e:
            print(f"\n❌ API error: {e}\n")
            history.pop()   # remove the user message we just added


if __name__ == "__main__":
    main()

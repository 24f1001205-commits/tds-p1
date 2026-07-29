#!/usr/bin/env python3
"""
Telegram data-analyst bot.

Listens for messages, asks an LLM (via aipipe.org) to work out the answer,
and replies with exactly one JSON object: {"answer": ..., "log_url": ...}

Required environment variables:
    TELEGRAM_BOT_TOKEN   - from @BotFather
    AIPIPE_TOKEN         - from aipipe.org/login
    LOG_URL              - public wget-able URL where run.jsonl will be hosted
                            (the raw GitHub URL, see below)

Optional:
    AIPIPE_MODEL         - default "gpt-5-mini"
    LOG_FILE             - default "run.jsonl"

Log auto-push (GitHub Contents API - works on ephemeral hosts like Render):
    GIT_AUTO_PUSH        - "true" to enable
    GITHUB_TOKEN         - fine-grained PAT, "Contents: read and write" on the repo
    GITHUB_REPO          - "owner/repo", e.g. "24f1001205-commits/tds-p1"
    GITHUB_BRANCH        - default "main"
"""
import asyncio
import base64
import json
import os
import time
import logging

from dotenv import load_dotenv
load_dotenv()  # no-op if there's no .env file (e.g. on Render, which uses real env vars)

import requests
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LOG_URL = os.environ["LOG_URL"]

MODEL = os.environ.get("AIPIPE_MODEL", "gpt-5-mini")
LOG_FILE = os.environ.get("LOG_FILE", "run.jsonl")
HISTORY_TURNS = 6  # how many past messages to keep per chat, for multi-turn context

# --- optional: auto-push run.jsonl to GitHub after every message ---
# Uses the GitHub Contents API directly (not local git commands), so it works
# even on hosts like Render where the filesystem is ephemeral and there's no
# persistent .git checkout or cached credentials to rely on.
GIT_AUTO_PUSH = os.environ.get("GIT_AUTO_PUSH", "false").lower() == "true"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")       # "owner/repo"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOG_FILE}"


def _push_log_via_api_sync() -> None:
    """Runs in a worker thread (see push_log) so it never blocks the bot
    from replying within the grader's time budget."""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }

        with open(LOG_FILE, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()

        # Need the current file's sha to update an existing file (None -> creates it)
        get_resp = requests.get(GITHUB_API_URL, headers=headers,
                                 params={"ref": GITHUB_BRANCH}, timeout=15)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        payload = {
            "message": "auto: update run.jsonl",
            "content": content_b64,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(GITHUB_API_URL, headers=headers, json=payload, timeout=15)
        if put_resp.status_code not in (200, 201):
            log.error("git auto-push failed: %s %s", put_resp.status_code, put_resp.text)
    except Exception:
        log.exception("git auto-push failed (log is still saved locally in %s)", LOG_FILE)


async def push_log() -> None:
    if GIT_AUTO_PUSH:
        await asyncio.to_thread(_push_log_via_api_sync)


client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)

conversation_history: dict[int, list[dict]] = {}

SYSTEM_PROMPT = (
    "You are a careful data analyst. The user's LAST message asks a data-analysis "
    "question and specifies exactly what JSON shape the final reply must have. "
    "Work out the real answer using public data you know (e.g. MOSPI statistics) "
    "or arithmetic on numbers given in the message. "
    "Reply with ONLY the raw JSON object the message asked for - no explanation, "
    "no markdown, no code fences, nothing before or after it."
)


def log_event(event: dict) -> None:
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def extract_json(text: str) -> dict:
    """Parse a JSON object out of the model's reply, tolerating stray text
    or markdown fences around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(text[start:end + 1])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text or ""
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-HISTORY_TURNS:],
        )
        raw_reply = response.choices[0].message.content.strip()
        parsed = extract_json(raw_reply)
    except Exception as e:
        log.exception("failed to get/parse a model reply")
        log_event({"type": "error", "chat_id": chat_id, "error": str(e)})
        parsed = {"answer": None}  # never leave the grader without valid JSON

    # Rebuild the final reply ourselves so it ALWAYS has exactly these two
    # keys, no matter what extra text/keys the model produced. Grading is
    # exact-match, so stray keys can break an otherwise-correct answer.
    final = {"answer": parsed.get("answer"), "log_url": LOG_URL}
    final_text = json.dumps(final)

    history.append({"role": "assistant", "content": final_text})
    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_text})

    await update.message.reply_text(final_text)

    # Push AFTER replying, and as a background task, so a slow/failed push
    # never delays the answer or eats into the grader's timeout budget.
    if GIT_AUTO_PUSH:
        asyncio.create_task(push_log())


def main() -> None:
    if GIT_AUTO_PUSH and not (GITHUB_TOKEN and GITHUB_REPO):
        log.warning("GIT_AUTO_PUSH is true but GITHUB_TOKEN/GITHUB_REPO not both set - "
                     "log pushes will fail until you set them.")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
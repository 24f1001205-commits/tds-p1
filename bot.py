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

# IMPORTANT: the question's own example JSON is illustrative only, and its
# shape varies per question. The example may or may not already show an
# "answer" key wrapping the payload (e.g. sometimes {"answer": {"state": "..."},
# "log_url": "..."}, sometimes just {"state": "..."}, "log_url": "..."} with no
# "answer" key at all). Our reply envelope is ALWAYS {"answer": ..., "log_url":
# ...} regardless of how the question's example is written - so the model must
# always end up wrapping its computed value under a single top-level "answer"
# key, no more and no less. Getting this wrong (double-wrapping, or not
# wrapping at all) silently breaks an otherwise-correct answer under exact-
# match grading, so the prompt below spells the rule out with explicit
# worked examples rather than leaving it to inference. We also enforce this
# in code below (normalize_answer) as a fallback in case the model doesn't
# comply.
SYSTEM_PROMPT = (
    "You are a careful data analyst replying to an automated grader over Telegram. "
    "The user's LAST message asks a data-analysis question and ends with a JSON "
    "example that tells you the exact shape to use. Your job has two independent parts:\n\n"
    "1. FIGURE OUT THE ANSWER VALUE.\n"
    "   Look at the JSON example in the message and find what belongs under its "
    "\"answer\" key (or, if the example has no \"answer\" key at all, treat the "
    "whole example - minus any \"log_url\" field - as the answer value).\n"
    "   This is the ONE piece you must compute yourself from public data "
    "(e.g. MOSPI statistics) or arithmetic on numbers given in the message. "
    "It can be a number, a string, an object, a list - whatever the example shows "
    "in that position, with placeholder text like \"<state name>\" replaced by your "
    "real computed answer.\n\n"
    "2. WRAP IT EXACTLY ONCE.\n"
    "   Your reply must be ONLY: {\"answer\": <the value from step 1>}\n"
    "   Do NOT wrap it twice. Do NOT nest another \"answer\" key inside it. "
    "The word \"answer\" must appear exactly once in your reply, as the single "
    "top-level key.\n\n"
    "Worked example - message ends with:\n"
    '  {"answer": {"state": "<state name>"}, "log_url": "<...>"}\n'
    "The answer value is whatever goes under \"answer\", i.e. {\"state\": \"...\"}. "
    "Correct reply: {\"answer\": {\"state\": \"Assam\"}}\n"
    'WRONG reply (double-wrapped): {"answer": {"answer": {"state": "Assam"}}}\n'
    'WRONG reply (unwrapped): {"state": "Assam"}\n\n'
    "Another worked example - message ends with:\n"
    '  {"state": "<state name>", "log_url": "<...>"}   (no \"answer\" key shown at all)\n'
    "Here the whole shown object (minus log_url) IS the answer value. "
    "Correct reply: {\"answer\": {\"state\": \"Assam\"}}\n\n"
    "Another worked example - message ends with:\n"
    '  {"values": [<numbers>]}   (a list inside a named key, no \"answer\" key, no log_url shown)\n'
    "The key name (here \"values\") is NOT \"answer\" and must be preserved exactly as shown - "
    "do not drop it and reply with a bare list. "
    "Correct reply: {\"answer\": {\"values\": [10.2, 20.4, 30.6]}}\n"
    'WRONG reply (dropped the wrapper key): {"answer": [10.2, 20.4, 30.6]}\n\n'
    "Never include \"log_url\" yourself - that is added by the system afterwards. "
    "Reply with ONLY the JSON object from step 2 - no explanation, no markdown, "
    "no code fences, nothing before or after it."
)


def log_event(event: dict) -> None:
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def extract_json(text: str):
    """Parse a JSON value out of the model's reply, tolerating stray text
    or markdown fences around it. Normally the reply is an object
    ({"answer": ...}), but if the model slips and replies with a bare
    array (a real risk for list-shaped answers, e.g. {"values": [...]}
    collapsed into just [...]), recover that too instead of raising and
    silently losing the answer to the outer except-block's answer_value=None.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates = []
    obj_start, obj_end = text.find("{"), text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])
    arr_start, arr_end = text.find("["), text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no valid JSON object or array found in reply", text, 0)


def normalize_answer(parsed: dict):
    """The model is instructed to always wrap its result in {"answer": ...},
    but if it slips and returns a bare payload instead (e.g. {"state": "Bihar"}
    with no "answer" key - which happens especially when the question's own
    example JSON doesn't show an "answer" key), don't silently lose it to a
    missing-key null. Treat the whole parsed object as the answer in that case.

    Also guard against the opposite slip: accidental double-wrapping, e.g.
    {"answer": {"answer": {"state": "Bihar"}}}. If the value under "answer"
    is itself a dict containing only a single "answer" key, unwrap one level -
    this is never a legitimate answer shape on its own, only a wrapping
    mistake."""
    if isinstance(parsed, dict) and "answer" in parsed:
        value = parsed["answer"]
        if isinstance(value, dict) and list(value.keys()) == ["answer"]:
            value = value["answer"]
        return value
    return parsed


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
        answer_value = normalize_answer(parsed)
    except Exception as e:
        log.exception("failed to get/parse a model reply")
        log_event({"type": "error", "chat_id": chat_id, "error": str(e)})
        answer_value = None  # never leave the grader without valid JSON

    # Rebuild the final reply ourselves so it ALWAYS has exactly these two
    # keys, no matter what extra text/keys the model produced. Grading is
    # exact-match, so stray keys can break an otherwise-correct answer.
    final = {"answer": answer_value, "log_url": LOG_URL}
    final_text = json.dumps(final)

    history.append({"role": "assistant", "content": final_text})
    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_text})

    await update.message.reply_text(final_text)

    # Push AFTER replying, and as a background task, so a slow/failed push
    # never delays the answer or eats into the grader's timeout budget.
    if GIT_AUTO_PUSH:
        asyncio.create_task(push_log())


def _start_dummy_webserver() -> None:
    """Render's free tier only offers Web Services, which require something
    listening on $PORT - our bot doesn't naturally do that since it just
    polls Telegram. This satisfies Render's port check with a no-op HTTP
    server running in a background thread, so the real bot loop below is
    unaffected. Not needed on a real Background Worker plan or another host."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass  # keep Render's log output clean of health-check noise

    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Dummy webserver listening on port %d (for Render's port check)", port)


def main() -> None:
    if GIT_AUTO_PUSH and not (GITHUB_TOKEN and GITHUB_REPO):
        log.warning("GIT_AUTO_PUSH is true but GITHUB_TOKEN/GITHUB_REPO not both set - "
                     "log pushes will fail until you set them.")
    if os.environ.get("PORT"):
        _start_dummy_webserver()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot starting (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

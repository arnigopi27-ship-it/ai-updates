import os
import re
import html
import json
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

# =========================
# CONFIG
# =========================

FEEDS = [
    ("OpenAI", "https://openai.com/news/rss.xml"),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("VentureBeat", "https://feeds.venturebeat.com/VentureBeat"),

    ("Google News - AI launches", "https://news.google.com/rss/search?q=%28AI+OR+LLM+OR+%22foundation+model%22+OR+multimodal+OR+agent%29+%28launch+OR+release+OR+update+OR+announced%29+when:1d&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Google News - AI companies", "https://news.google.com/rss/search?q=%28OpenAI+OR+Anthropic+OR+Google+DeepMind+OR+xAI+OR+Meta+AI+OR+Mistral+OR+Moonshot+AI+OR+DeepSeek+OR+Cohere+OR+Perplexity%29+when:1d&hl=en-IN&gl=IN&ceid=IN:en")
]

KEYWORDS = [
    "ai", "artificial intelligence", "llm", "model", "agent", "agents",
    "multimodal", "reasoning", "inference", "open source", "release",
    "launch", "launched", "announced", "announcement", "update", "upgrades",
    "openai", "anthropic", "google deepmind", "deepmind", "gemini",
    "meta ai", "xai", "grok", "mistral", "moonshot", "kimi",
    "deepseek", "cohere", "perplexity", "claude", "gpt"
]

BAD_PATTERNS = [
    "job", "hiring", "career", "course", "tutorial", "webinar",
    "coupon", "discount", "sponsored", "affiliate"
]

HOURS_BACK = 24
MAX_ITEMS_BEFORE_SUMMARY = 12
MAX_ITEMS_FOR_GEMINI = 6

CLIQ_WEBHOOK_URL = os.environ["CLIQ_WEBHOOK_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# =========================
# HELPERS
# =========================

def clean_text(text):
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_title(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()

def parse_entry_date(entry):
    for key in ("published", "updated", "created"):
        if key in entry:
            try:
                return parsedate_to_datetime(entry[key]).astimezone(timezone.utc)
            except Exception:
                pass
    return None

def looks_relevant(title, summary=""):
    hay = f"{title} {summary}".lower()
    if any(bad in hay for bad in BAD_PATTERNS):
        return False
    return any(keyword in hay for keyword in KEYWORDS)

def fetch_feed_items():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    items = []
    seen_links = set()
    seen_titles = set()

    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for entry in getattr(feed, "entries", []):
            title = clean_text(entry.get("title", "No title"))
            summary = clean_text(entry.get("summary", ""))
            link = entry.get("link", "").strip()
            published = parse_entry_date(entry) or datetime.now(timezone.utc)

            if not link:
                continue
            if published < cutoff:
                continue
            if not looks_relevant(title, summary):
                continue

            norm_title = normalize_title(title)
            if link in seen_links or norm_title in seen_titles:
                continue

            seen_links.add(link)
            seen_titles.add(norm_title)

            items.append({
                "source": source,
                "title": title,
                "summary": summary[:500],
                "link": link,
                "published": published
            })

    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:MAX_ITEMS_BEFORE_SUMMARY]

def build_gemini_prompt(items):
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. Title: {item['title']}\n"
            f"   Source: {item['source']}\n"
            f"   Summary: {item['summary']}\n"
            f"   Link: {item['link']}"
        )

    item_block = "\n\n".join(lines)

    prompt = f"""
You are preparing a concise daily AI news digest for a Zoho Cliq channel.

Task:
- Read the news items below.
- Select the most important items.
- Remove duplicates or near-duplicates.
- Focus only on actual AI news such as model releases, major updates, product launches, company announcements, research, funding, regulation, or notable partnerships.
- Ignore generic opinion pieces, job posts, tutorials, and low-value marketing posts.

Output rules:
- Return plain text only.
- Keep it short and clean for a team chat.
- Start with: AI News Digest - {datetime.now(timezone.utc).strftime("%d %b %Y")}
- Then add 4 to 6 bullet points.
- Each bullet must contain:
  - short headline rewritten in simple words
  - why it matters in one short phrase
  - source name
- After bullets, add a section called "Links:" and list the selected article URLs, one per line.
- Do not use markdown tables.
- Do not invent facts.
- If the items are weak, still return the best available ones.

News items:
{item_block}
""".strip()

    return prompt

def summarize_with_gemini(items):
    selected = items[:MAX_ITEMS_FOR_GEMINI]
    prompt = build_gemini_prompt(selected)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No Gemini candidates returned")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        raise ValueError("Empty Gemini response")

    return text

def fallback_digest(items):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")

    if not items:
        return f"AI News Digest - {today}\n\nNo strong AI news found in the last {HOURS_BACK} hours."

    lines = [f"AI News Digest - {today}", ""]
    for item in items[:6]:
        lines.append(f"- {item['title']} ({item['source']})")
    lines.append("")
    lines.append("Links:")
    for item in items[:6]:
        lines.append(item["link"])
    return "\n".join(lines)

def post_to_cliq(message):
    payload = {
        "text": message
    }
    resp = requests.post(CLIQ_WEBHOOK_URL, json=payload, timeout=30)
    resp.raise_for_status()

# =========================
# MAIN
# =========================

def main():
    items = fetch_feed_items()

    if not items:
        message = fallback_digest([])
        post_to_cliq(message)
        return

    try:
        message = summarize_with_gemini(items)
    except Exception:
        message = fallback_digest(items)

    post_to_cliq(message)

if __name__ == "__main__":
    main()

import os
import re
import html
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

FEEDS = [
    ("OpenAI", "https://openai.com/news/rss.xml"),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("VentureBeat", "https://feeds.venturebeat.com/VentureBeat"),
]

KEYWORDS = [
    "ai", "artificial intelligence", "llm", "model", "agent",
    "multimodal", "release", "launch", "announced",
    "openai", "anthropic", "gemini", "claude", "gpt", "deepseek"
]

HOURS_BACK = 24
MAX_ITEMS = 6

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_PAGE_ID = os.environ["NOTION_PAGE_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def clean_text(text):
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
    return any(keyword in hay for keyword in KEYWORDS)

def fetch_feed_items():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    items = []
    seen_links = set()

    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for entry in getattr(feed, "entries", []):
            title = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", ""))
            link = entry.get("link", "").strip()
            published = parse_entry_date(entry) or datetime.now(timezone.utc)

            if not link or link in seen_links:
                continue
            if published < cutoff:
                continue
            if not looks_relevant(title, summary):
                continue

            seen_links.add(link)
            items.append({
                "source": source,
                "title": title,
                "summary": summary[:300],
                "link": link,
                "published": published
            })

    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:MAX_ITEMS]

def summarize_with_gemini(items):
    lines = []
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item['title']} ({item['source']})\n   {item['summary']}")

    prompt = f"""
You are preparing a daily AI news digest for a tech team.
Summarize these news items clearly with emoji.
Format: numbered list. Keep each point 1-2 lines.
Date: {datetime.now(timezone.utc).strftime("%d %b %Y")}

News:
{chr(10).join(lines)}
""".strip()

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()

def post_to_notion(summary, items):
    today = datetime.now(timezone.utc).strftime("%d %b %Y")

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    blocks = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": f"🤖 AI News - {today}"}}]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary}}]
            }
        },
        {
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": "📰 Source Links"}}]
            }
        }
    ]

    for item in items:
        blocks.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"{item['title']} ({item['source']})",
                            "link": {"url": item['link']}
                        }
                    }
                ]
            }
        })

    blocks.append({"object": "block", "type": "divider", "divider": {}})

    url = f"https://api.notion.com/v1/blocks/{NOTION_PAGE_ID}/children"
    resp = requests.patch(url, headers=headers, json={"children": blocks}, timeout=30)
    print(f"Notion status: {resp.status_code}")
    resp.raise_for_status()

def main():
    print("Fetching feed items...")
    items = fetch_feed_items()
    print(f"Found {len(items)} items")

    if not items:
        summary = f"No AI news found today ({datetime.now(timezone.utc).strftime('%d %b %Y')})"
    else:
        try:
            print("Summarizing with Gemini...")
            summary = summarize_with_gemini(items)
        except Exception as e:
            print(f"Gemini failed: {e}")
            summary = "\n".join([f"• {item['title']} ({item['source']})" for item in items])

    print("Posting to Notion...")
    post_to_notion(summary, items)
    print("Done! ✅")

if __name__ == "__main__":
    main()

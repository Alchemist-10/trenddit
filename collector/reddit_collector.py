# collector/reddit_collector.py
"""
Reddit collector: fetch latest posts matching a keyword and insert into Supabase.

Usage:
  python collector/reddit_collector.py --keyword "openai"
"""

import os
import argparse
import time
from datetime import datetime, timezone
import praw
from supabase import create_client
from nlp.sentiment import analyze_sentiment
from nlp.embeddings import embed_text
import json

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "trenddit/0.1 by demo")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_and_store(keyword, limit=100):
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        raise RuntimeError("Missing REDDIT_CLIENT_ID/SECRET env vars")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    # Search across r/all
    query = keyword
    posts_fetched = []
    for submission in reddit.subreddit("all").search(query, limit=limit, sort="new"):
        data = {
            "id": f"reddit:{submission.id}",
            "source": "reddit",
            "source_id": submission.id,
            "keyword": keyword,
            "title": submission.title,
            "body": submission.selftext,
            "author": str(submission.author) if submission.author else None,
            "url": submission.url,
            "score": submission.score,
            "created_at": datetime.fromtimestamp(
                submission.created_utc, tz=timezone.utc
            ).isoformat(),
            "metadata": json.dumps(
                {
                    "subreddit": submission.subreddit.display_name,
                    "num_comments": submission.num_comments,
                }
            ),
        }
        # deduplicate: try insert, if conflict ignore (use upsert with on_conflict not available in supabase-py currently)
        posts_fetched.append(data)

    # preprocess: compute sentiment and embedding for each
    for p in posts_fetched:
        text = (p.get("title") or "") + "\n" + (p.get("body") or "")
        score, label = analyze_sentiment(text)
        embedding = embed_text(text)  # returns list or numpy array
        p["sentiment_score"] = float(score)
        p["sentiment_label"] = label
        # Supabase expects JSON for metadata column but we stored as string earlier; ensure format
        try:
            p["metadata"] = json.loads(p["metadata"])
        except:
            p["metadata"] = {}
        # embedding column: supabase-py will attempt to map python list to Postgres vector/dbl[] if configured
        p["embedding"] = (
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

        # Insert, ignoring conflict manually: do a select first to avoid dup insert
        existing = supabase.table("posts").select("id").eq("id", p["id"]).execute()
        if existing.data:
            print(f"Skipping existing post {p['id']}")
            continue
        res = supabase.table("posts").insert(p).execute()
        if res.error:
            print("Insert error:", res.error)
        else:
            print("Inserted", p["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    fetch_and_store(args.keyword, args.limit)

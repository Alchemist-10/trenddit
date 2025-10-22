# collector/reddit_collector.py
"""
Reddit collector: fetch latest posts matching a keyword and insert into Supabase.

Usage:
  python -m collector.reddit_collector --keyword "openai"
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
from dotenv import load_dotenv

load_dotenv()
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
    print(f"Fetching {limit} posts from Reddit for '{keyword}'...")
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
        posts_fetched.append(data)

    if not posts_fetched:
        print("No new posts found on Reddit.")
        return

    print(f"Found {len(posts_fetched)} posts. Processing NLP...")

    # 1. Pre-process ALL posts first (NLP, embeddings)
    for p in posts_fetched:
        text = (p.get("title") or "") + "\n" + (p.get("body") or "")
        score, label = analyze_sentiment(text)
        embedding = embed_text(text)  # returns list or numpy array

        p["sentiment_score"] = float(score)
        p["sentiment_label"] = label

        # Supabase expects JSON for metadata column
        try:
            p["metadata"] = json.loads(p["metadata"])
        except:
            p["metadata"] = {}

        # embedding column
        p["embedding"] = (
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

    # --- Efficient Batch Processing ---

    # 2. Get all IDs to check for duplicates
    ids_to_check = [p["id"] for p in posts_fetched]
    posts_to_insert = []

    try:
        # 3. Check for duplicates in ONE database call
        existing_res = (
            supabase.table("posts").select("id").in_("id", ids_to_check).execute()
        )
        existing_ids = {p["id"] for p in existing_res.data}

        # 4. Filter out any posts we already have
        for p in posts_fetched:
            if p["id"] not in existing_ids:
                posts_to_insert.append(p)
            else:
                print(f"Skipping existing post {p['id']}")

        # 5. Insert all new posts in ONE database call
        if posts_to_insert:
            print(f"Inserting {len(posts_to_insert)} new posts...")
            supabase.table("posts").insert(posts_to_insert).execute()
            print("Batch insert complete.")

            # 6. OPTIONAL: also populate separate embeddings table if present
            try:
                emb_rows = [
                    {"post_id": p["id"], "vector": p.get("embedding")}
                    for p in posts_to_insert
                    if p.get("embedding") is not None
                ]
                if emb_rows:
                    supabase.table("embeddings").upsert(emb_rows).execute()
                    print("Embeddings table upsert complete.")
            except Exception as e:
                # Non-fatal: keep going even if embeddings table not present
                print(f"Note: couldn't upsert into embeddings table: {e}")
        else:
            print("All posts found were already in the database.")

    except Exception as e:
        print(f"Error during batch database operation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    fetch_and_store(args.keyword, args.limit)

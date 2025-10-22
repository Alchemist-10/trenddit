# app/streamlit_app.py
"""
Streamlit app for Trenddit prototype.

Run:
  streamlit run app/streamlit_app.py

Environment variables expected (store securely, not in frontend/public repos):
- SUPABASE_URL
- SUPABASE_KEY   (service_role key if you need insert access; for user-only actions use anon + Edge functions)
"""


import os
import time
from dotenv import load_dotenv

# Load .env file at the very beginning
load_dotenv()

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from supabase import create_client, Client
from sklearn.feature_extraction.text import CountVectorizer
from io import StringIO

# local utils
from utils import pretty_time_ago

# ---- configure ----
st.set_page_config(page_title="Trenddit — Prototype", layout="wide")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment. See README.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- UI ----
st.title("Trenddit — Real-time Reddit + X trend analyzer (Prototype)")

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input("Search keyword", value="openai")
with col2:
    source = st.multiselect(
        "Source", options=["reddit", "twitter"], default=["reddit", "twitter"]
    )
timeframe = st.selectbox("Timeframe", ["1h", "6h", "24h", "7d", "30d"], index=2)
refresh_mode = st.radio("Live update mode", ["polling", "realtime (optional)"], index=0)
poll_interval = st.slider(
    "Polling interval (seconds)", min_value=10, max_value=300, value=30
)

# Save query
if st.button("Save query"):
    try:
        supabase.table("queries").insert(
            {"keyword": keyword, "sources": source}
        ).execute()
        st.success("Saved query")
    except Exception as e:
        st.error(f"Failed to save query: {e}")

# timeframe -> start_time
now = datetime.utcnow()
tf_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}
hours = tf_map.get(timeframe, 24)
start_time = now - timedelta(hours=hours)


# ---- fetch posts from supabase ----
def fetch_posts(keyword, sources, start_time, limit=1000, offset=0):
    # Simple RPC via PostgREST style
    query = supabase.table("posts").select("*").order("created_at", desc=True)
    query = query.filter("created_at", "gte", start_time.isoformat())
    if keyword:
        # crude match: keyword in title or body or posts.keyword equals
        query = query.filter("keyword", "eq", keyword)
    if sources:
        # **FIXED**: Supabase 'in' filter expects a list or tuple
        query = query.filter("source", "in", tuple(sources))

    try:
        # Pagination for stream
        res = query.range(offset, offset + limit - 1).execute()
        # In v2, if successful, res.data contains the list
        return res.data or []
    except Exception as e:
        # Handle the exception
        st.error(f"Error fetching posts: {e}")
        return []


# helper to convert to dataframe
def posts_to_df(posts):
    df = pd.DataFrame(posts)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["inserted_at"] = pd.to_datetime(df["inserted_at"])
    return df


# initial load
posts = fetch_posts(keyword, source, start_time, limit=500)
df = posts_to_df(posts)

# ---- Layout: Left: Charts, Right: Stream -->
left, right = st.columns([2, 1])

with left:
    # Sentiment timeline: compute hourly average and volume
    st.subheader("Sentiment timeline")
    if df.empty:
        st.info("No posts found for this query/timeframe.")
    else:
        df = df.sort_values("created_at")
        # ensure numeric
        df["sentiment_score"] = pd.to_numeric(
            df["sentiment_score"], errors="coerce"
        ).fillna(0)
        df.set_index("created_at", inplace=True)
        # resample depending on timeframe
        if hours <= 24:
            res = df["sentiment_score"].resample("15min").mean().fillna(method="ffill")
            vol = df["sentiment_score"].resample("15min").count()
        else:
            res = df["sentiment_score"].resample("1H").mean().fillna(method="ffill")
            vol = df["sentiment_score"].resample("1H").count()
        timeline = pd.DataFrame({"avg_sentiment": res, "volume": vol}).reset_index()
        fig = px.line(
            timeline,
            x="created_at",
            y="avg_sentiment",
            title=f"Average sentiment for '{keyword}'",
        )
        fig.add_bar(
            x=timeline["created_at"],
            y=timeline["volume"],
            name="volume",
            opacity=0.4,
            yaxis="y2",
        )
        # add secondary axis
        fig.update_layout(
            yaxis=dict(title="Avg sentiment"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top n-grams
    st.subheader("Top keywords / n-grams")
    if not df.empty:
        # combine title+body safely
        if "title" in df.columns and "body" in df.columns:
            df_text = df["title"].fillna("") + " " + df["body"].fillna("")
        elif "title" in df.columns:
            df_text = df["title"].fillna("")
        elif "body" in df.columns:
            df_text = df["body"].fillna("")
        else:
            df_text = pd.Series([""])

        # extract top n-grams
        vectorizer = CountVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=30
        )
        X = vectorizer.fit_transform(df_text)
        freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
        freq_df = pd.DataFrame(
            sorted(freqs, key=lambda x: x[1], reverse=True), columns=["term", "count"]
        )
        st.dataframe(freq_df.head(25))
    else:
        st.write("No text to analyze.")

    # Topic clusters (basic)
    st.subheader("Topic clusters (approximate)")
    # we will show top cluster labels and representative post
    if not df.empty and "embedding" in df.columns:
        # embeddings stored as lists/dicts; convert to numpy
        try:
            import ast

            emb_list = df["embedding"].apply(
                lambda x: (
                    np.array(x)
                    if isinstance(x, list)
                    else (
                        np.array(ast.literal_eval(x))
                        if isinstance(x, str) and x.startswith("[")
                        else np.array([0] * 384)  # Fallback for None or invalid
                    )
                )
            )

            # simple KMeans
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans

            E = np.vstack(emb_list.to_list())
            n_clusters = min(
                6, max(2, E.shape[0] // 10)
            )  # Ensure at least 2 clusters if possible

            if n_clusters < 2:
                st.write("Not enough data to form clusters.")
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(
                    E
                )
                df["cluster"] = kmeans.labels_
                cluster_summary = []

                # Make sure df index is reset for iloc to work as expected
                df_reset = df.reset_index()

                for c in sorted(df["cluster"].unique()):
                    subset = df_reset[df_reset["cluster"] == c]
                    if not subset.empty:
                        rep = subset.iloc[0]  # Get first post in cluster
                        cluster_summary.append(
                            (
                                c,
                                len(subset),
                                rep.get("title") or rep.get("body", "")[:200],
                                rep.get("url"),
                            )
                        )

                for c, count, rep_text, url in cluster_summary:
                    st.markdown(f"**Cluster {c}** — {count} posts")
                    st.write(rep_text)
                    if url:
                        st.write(url)

        except Exception as e:
            st.warning("Clustering failed: " + str(e))
    else:
        st.write("No embeddings available for clustering.")

with right:
    st.subheader("Live posts")
    page_size = int(st.number_input("Page size", min_value=5, max_value=100, value=10))
    # pagination controls
    page = int(st.number_input("Page", min_value=1, value=1))
    offset = (page - 1) * page_size
    posts_page = fetch_posts(
        keyword, source, start_time, limit=page_size, offset=offset
    )
    if not posts_page:
        st.write("No posts to show.")
    else:
        for p in posts_page:
            created = pretty_time_ago(p.get("created_at"))
            score = p.get("sentiment_score") or 0
            label = p.get("sentiment_label") or "neutral"
            st.markdown(f"**{p.get('title') or p.get('body', '')[:80]}** ")
            st.caption(
                f"{p.get('source')} • {p.get('author')} • {created} • sentiment: {score:.2f} ({label})"
            )
            st.write(p.get("body") or "")

    # Alerts panel
    st.subheader("Alerts")
    # **FIXED**: Use try/except for supabase-py v2 error handling
    try:
        alerts_res = (
            supabase.table("alerts")
            .select("*")
            .order("triggered_at", desc=True)
            .limit(10)
            .execute()
        )
        alerts = alerts_res.data or []
        if not alerts:
            st.write("No alerts")
        else:
            for a in alerts:
                st.warning(
                    f"[{a.get('alert_type')}] {a.get('message')} — {pretty_time_ago(a.get('triggered_at'))}"
                )
    except Exception as e:
        st.error(f"Error loading alerts: {e}")


# Export CSV
if st.button("Export CSV"):
    if df.empty:
        st.warning("No data to export")
    else:
        csv = df.reset_index().to_csv(index=False)
        st.download_button(
            "Download posts CSV",
            data=csv,
            file_name=f"trenddit_{keyword}_{now.date()}.csv",
            mime="text/csv",
        )

# Polling loop (optional)
if refresh_mode == "polling":
    st.info(f"Polling every {poll_interval}s. Click ⟳ to refresh now.")
    # Add a manual refresh button
    # **FIXED**: Use st.rerun() instead of st.experimental_rerun()
    if st.button("⟳ Refresh"):
        st.rerun()

    # simple auto-refresh
    if st.checkbox("Auto-refresh", value=False):
        # streamlit can't truly run background loops; we leverage rerun every poll_interval seconds
        time.sleep(poll_interval)
        # **FIXED**: Use st.rerun() instead of st.experimental_rerun()
        st.rerun()

# Note: For true realtime you can add supabase client-side realtime subscription using JS in React,
# or run a background thread on server to push updates to DB and use client polling or SSE.

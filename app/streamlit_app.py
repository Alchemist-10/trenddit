# app/streamlit_app.py


import os
import sys
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

# Allow importing project modules (collector) when running via Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # Directly call the collector from the UI when keyword is submitted
    from collector.reddit_collector import fetch_and_store as collect_reddit
except Exception as _e:
    collect_reddit = None  # Fallback if not available; we'll guard usage

# ---- configure ----
st.set_page_config(page_title="Trenddit — Prototype", layout="wide")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY in environment. See README.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- UI ----
# Global CSS + top navbar with smooth scrolling (CSS-only)
st.markdown(
    """
        <style>
            :root{
                --td-nav-height: 56px;
                --td-nav-bg: linear-gradient(90deg, #ffffff 0%, #f7fbff 50%, #ffffff 100%);
                --td-link: #1f77b4;
                --td-link-hover: #125a89;
            }
            html { scroll-behavior: smooth; }
            /* Offset anchors so sticky navbar doesn't cover headings */
            [id] { scroll-margin-top: calc(var(--td-nav-height) + 8px); }

            .td-navbar{
                position: sticky; top: 0; z-index: 9999;
                height: var(--td-nav-height);
                display: flex; align-items: center;
                padding: 0 12px;
                background: var(--td-nav-bg);
                backdrop-filter: blur(6px);
                border-bottom: 1px solid #eaecef;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-radius: 0 0 10px 10px;
            }
            .td-nav-inner{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
            .td-brand{ font-weight: 700; margin-right: 6px; color: #0f172a; opacity: .9; }
            .td-sep{ opacity: .25; margin: 0 6px; }
            .td-navbar a{
                display: inline-flex; align-items: center; gap: 6px;
                padding: 6px 10px; border-radius: 8px;
                color: var(--td-link); text-decoration: none; font-weight: 600; font-size: 0.92rem;
                transition: all .15s ease-in-out;
            }
            .td-navbar a:hover{ color: var(--td-link-hover); background: rgba(31,119,180,0.08); }
            .td-navbar a:active{ transform: translateY(1px); }
            .td-emoji{ font-size: 1.05rem; }
            @media (max-width: 520px){ .td-brand, .td-sep { display:none; } }
        </style>
        <nav class=\"td-navbar\">
            <div class=\"td-nav-inner\">
                <span class=\"td-brand\">Navigate</span>
                <span class=\"td-sep\">·</span>
                <a href=\"#sentiment-timeline\"><span class=\"td-emoji\">📈</span><span>Sentiment</span></a>
                <a href=\"#top-ngrams\"><span class=\"td-emoji\">🧩</span><span>Top keywords</span></a>
                <a href=\"#topic-clusters\"><span class=\"td-emoji\">🗂️</span><span>Clusters</span></a>
                <a href=\"#live-posts\"><span class=\"td-emoji\">⚡</span><span>Live posts</span></a>
                <a href=\"#alerts\"><span class=\"td-emoji\">🚨</span><span>Alerts</span></a>
            </div>
        </nav>
        """,
    unsafe_allow_html=True,
)

st.title("Trenddit — Real-time Reddit + X trend analyzer (Prototype)")

col1, col2 = st.columns([3, 1])
with col1:
    # Submit on Enter: set a flag via on_change, then process below
    def _on_keyword_submit():
        kw = st.session_state.get("keyword", "").strip()
        if kw:
            st.session_state["should_collect"] = True

    keyword = st.text_input(
        "Search keyword",
        value="openai",
        key="keyword",
        on_change=_on_keyword_submit,
        placeholder="e.g., openai, donald trump, cricket, ronaldo",
    )
with col2:
    source = st.multiselect(
        "Source", options=["reddit", "twitter"], default=["reddit", "twitter"]
    )
timeframe = st.selectbox("Timeframe", ["1h", "6h", "24h", "7d", "30d"], index=2)
refresh_mode = st.radio("Live update mode", ["polling", "realtime (optional)"], index=0)
poll_interval = st.slider(
    "Polling interval (seconds)", min_value=10, max_value=300, value=30
)

# If user hit Enter in the keyword box, run collector once before fetching
if st.session_state.get("should_collect"):
    if collect_reddit is not None:
        try:
            with st.spinner(
                f"Collecting posts for '{st.session_state.get('keyword','')}'..."
            ):
                collect_reddit(st.session_state.get("keyword", "").strip(), limit=100)
            st.toast("Collection complete.")
        except Exception as e:
            st.warning(f"Collector error: {e}")
    else:
        st.info(
            "Collector module not available in this session; skipping auto-collect."
        )
    # Reset flag and proceed to fetch
    st.session_state["should_collect"] = False
    # Give DB a brief moment to commit
    time.sleep(0.5)

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
# Use timezone-aware UTC to avoid off-by-timezone filters in PostgREST
from datetime import timezone as _tz

now = datetime.now(_tz.utc)
tf_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}
hours = tf_map.get(timeframe, 24)
start_time = now - timedelta(hours=hours)


# ---- fetch posts from supabase ----
def fetch_posts(keyword, sources, start_time, limit=1000, offset=0):
    # Simple RPC via PostgREST style
    query = supabase.table("posts").select("*").order("created_at", desc=True)
    # Ensure ISO string includes timezone (Z) for timestamptz column
    query = query.filter("created_at", "gte", start_time.isoformat())

    if keyword:
        kw = keyword.strip()
        # Broaden search: title/body contains keyword OR keyword column matches (case-insensitive)
        # PostgREST OR syntax: or=(col.op.val,...) — supabase-py v2 exposes .or_
        try:
            query = query.or_(
                f"title.ilike.%{kw}%,body.ilike.%{kw}%,keyword.ilike.%{kw}%"
            )
        except Exception:
            # Fallback to equality on keyword if .or_ not available
            query = query.filter("keyword", "eq", kw)

    if sources:
        # Use native in_ helper for reliability
        try:
            query = query.in_("source", list(sources))
        except Exception:
            query = query.filter("source", "in", list(sources))

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
    st.markdown('<a id="sentiment-timeline"></a>', unsafe_allow_html=True)
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
    st.markdown('<a id="top-ngrams"></a>', unsafe_allow_html=True)
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
    st.markdown('<a id="topic-clusters"></a>', unsafe_allow_html=True)
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
    st.markdown('<a id="live-posts"></a>', unsafe_allow_html=True)
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
    st.markdown('<a id="alerts"></a>', unsafe_allow_html=True)
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

# app/streamlit_app.py


import os
import sys
import time
import re
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

# ===========================
# AUTHENTICATION LOGIC
# ===========================


def sign_up(email: str, password: str):
    """Sign up a new user with Supabase Auth"""
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        return response
    except Exception as e:
        return {"error": str(e)}


def sign_in(email: str, password: str):
    """Sign in an existing user"""
    try:
        response = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        # Return the full response object
        return response
    except Exception as e:
        return {"error": str(e)}


def sign_out():
    """Sign out current user"""
    try:
        supabase.auth.sign_out()
        return True
    except Exception as e:
        st.error(f"Sign out error: {e}")
        return False


def check_authentication():
    """Check if user is authenticated using Supabase session"""
    try:
        # Get current session from Supabase
        session = supabase.auth.get_session()

        # Check if session exists and has a user
        if session and hasattr(session, "user") and session.user:
            # User is authenticated via Supabase
            st.session_state.authenticated = True
            st.session_state.user = session.user
            return True
        # Alternative: check if session_state already has user from recent login
        elif st.session_state.get("authenticated") and st.session_state.get("user"):
            return True
        else:
            # No valid session
            st.session_state.authenticated = False
            st.session_state.user = None
            return False
    except Exception as e:
        # If session check fails, check session_state as fallback
        if st.session_state.get("authenticated") and st.session_state.get("user"):
            return True
        # Otherwise assume not authenticated
        st.session_state.authenticated = False
        st.session_state.user = None
        return False


# Initialize session state on first load
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

# Check authentication status on every page load
check_authentication()


# ===========================
# TEXT SANITIZATION HELPERS
# ===========================
def strip_html_tags(text: str) -> str:
    """Remove HTML tags and collapse extra whitespace from a string.
    Keeps plain text only so user content never injects markup into our templates.
    """
    if not text:
        return ""
    # Remove tags
    no_tags = re.sub(r"<[^>]+>", " ", str(text))
    # Collapse whitespace
    return re.sub(r"\s+", " ", no_tags).strip()


# ---- UI ----
# Modern CSS with design tokens, navbar, KPI cards, chips, skeleton loaders
st.markdown(
    """
    <style>
        :root{
            --td-accent: #7c3aed;
            --td-accent-2: #6366f1;
            --td-bg: #f8fafc;
            --td-card: #ffffff;
            --td-text: #0f172a;
            --td-muted: #6b7280;
            --td-border: #e2e8f0;
            --td-ring: rgba(124,58,237,.25);
            --td-nav-height: 64px;
        }
        @media (prefers-color-scheme: dark){
            :root{
                --td-bg: #0f172a;
                --td-card: #1e293b;
                --td-text: #f1f5f9;
                --td-muted: #94a3b8;
                --td-border: #334155;
            }
        }
        
        /* Smooth scrolling for all scrollable elements */
        html, body, * {
            scroll-behavior: smooth !important;
        }
        
        /* Target Streamlit's main container */
        section[data-testid="stAppViewContainer"],
        section[data-testid="stAppViewContainer"] > div,
        .main {
            scroll-behavior: smooth !important;
        }
        
        /* Offset for anchor links to account for fixed navbar */
        [id], a[id], [id]::before {
            scroll-margin-top: calc(var(--td-nav-height) + 30px);
            scroll-snap-margin-top: calc(var(--td-nav-height) + 30px);
        
        }
        
        body { 
            -webkit-font-smoothing: antialiased; 
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Modern fixed navbar with enhanced animations */
        .td-navbar{
            position: fixed !important;
            top: 0 !important; 
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            max-width: 100vw !important;
            z-index: 9999 !important;
            min-height: var(--td-nav-height);
            display: flex !important; 
            align-items: center; 
            justify-content: space-between;
            padding: 0 24px;
            background: linear-gradient(135deg, var(--td-accent) 0%, var(--td-accent-2) 100%);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-bottom: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.08);
            border-radius: 0 0 24px 24px;
            box-sizing: border-box;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin: 0 !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        /* Hardening: ensure navbar shows across desktop breakpoints with higher specificity */
        @media (min-width: 769px){
            body .td-navbar, .stApp .td-navbar, .stAppViewContainer .td-navbar{
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
        }
        @media (min-width: 1024px){
            body .td-navbar, .stApp .td-navbar, .stAppViewContainer .td-navbar{
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
        }
        /* Ensure navbar is visible on desktop/large screens as well */
        @media (min-width: 769px){
            .td-navbar{
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
        }
        
        /* Navbar hover effect */
        .td-navbar:hover {
            box-shadow: 0 6px 28px rgba(0,0,0,0.15), 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Push content below fixed navbar */
        .block-container {
            padding-top: calc(var(--td-nav-height) + 24px) !important;
        }

        /* Brand area with logo animation */
        .td-brand-area{
            display: flex !important; 
            align-items: center; 
            gap: 12px;
            transition: transform 0.3s ease;
            visibility: visible !important;
        }
        .td-brand-area:hover {
            transform: scale(1.05);
        }
        .td-logo{ 
            font-size: 1.6rem;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            display: inline-block !important;
        }
        .td-brand{
            font-weight: 700; 
            font-size: 1.15rem;
            color: white; 
            letter-spacing: -0.02em;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: inline-block !important;
        }
        
        /* Navigation links container */
        .td-nav-links{
            display: flex !important; 
            gap: 8px; 
            align-items: center; 
            flex-wrap: wrap;
            visibility: visible !important;
        }
        
        /* Modern pill-style navigation links with smooth animations */
        .td-nav-link{
            display: inline-flex !important; 
            align-items: center; 
            gap: 8px;
            padding: 11px 20px; 
            border-radius: 50px;
            color: rgba(255,255,255,0.95); 
            text-decoration: none;
            font-weight: 600; 
            font-size: 0.94rem;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
            position: relative;
            cursor: pointer;
            overflow: hidden;
            visibility: visible !important;
        }
        
        /* Shimmer effect on hover */
        .td-nav-link::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .td-nav-link:hover::before {
            left: 100%;
        }
        
        /* Hover state with enhanced lift effect */
        .td-nav-link:hover{
            color: white;
            background: rgba(255,255,255,0.28);
            border-color: rgba(255,255,255,0.4);
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2), 0 2px 8px rgba(0,0,0,0.15);
        }
        
        /* Active/click state */
        .td-nav-link:active{
            transform: translateY(-1px) scale(0.98);
            box-shadow: 0 3px 10px rgba(0,0,0,0.15);
            transition: all 0.1s ease;
        }
        
        /* Focus state for accessibility */
        .td-nav-link:focus {
            outline: 2px solid rgba(255,255,255,0.6);
            outline-offset: 2px;
        }
        
        .td-emoji{ 
            font-size: 1.1rem;
            transition: transform 0.3s ease;
        }
        
        .td-nav-link:hover .td-emoji {
            transform: scale(1.15) rotate(5deg);
        }
        
        /* Responsive design - Mobile overrides */
        @media (max-width: 768px){
            .td-navbar { 
                padding: 0 16px; 
                border-radius: 0 0 18px 18px; 
            }
            .td-nav-links{ gap: 6px; }
            .td-nav-link{ padding: 9px 14px; font-size: 0.9rem; }
            .td-logo { font-size: 1.5rem; }
        }
        
        /* Extra small screens */
        @media (max-width: 540px){
            .td-brand{ display: none !important; }
            .td-nav-links{ gap: 5px; }
            .td-nav-link{ padding: 8px 12px; font-size: 0.88rem; }
            .td-navbar { padding: 0 12px; border-radius: 0 0 16px 16px; }
            .td-logo { font-size: 1.4rem; }
        }

        /* --- ALL OTHER STYLES (KPI, Chips, etc.) --- */
        /* --- (These are unchanged from your original code) --- */
        
        /* KPI Cards */
        .td-kpi-row{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .td-kpi-card{
            background: var(--td-card);
            border: 1px solid var(--td-border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        }
        .td-kpi-card:hover{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .td-kpi-label{
            font-size: 0.8rem;
            color: var(--td-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .td-kpi-value{
            font-size: 2rem;
            font-weight: 700;
            color: var(--td-text);
            line-height: 1.2;
        }
        .td-kpi-icon{
            font-size: 1.5rem;
            opacity: 0.7;
        }
        .td-delta{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            margin-top: 8px;
            padding: 4px 8px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .td-delta.up{ background: #d1fae5; color: #065f46; }
        .td-delta.down{ background: #fee2e2; color: #991b1b; }

        /* Chips */
        .td-chip{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--td-card);
            border: 1px solid var(--td-border);
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--td-text);
            transition: all 0.2s ease;
            margin: 4px;
        }
        .td-chip:hover{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .td-chip-count{
            background: var(--td-accent);
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.75rem;
        }

        /* Post Cards */
        .td-post-card{
            background: var(--td-card);
            border: 1px solid var(--td-border);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        }
        .td-post-card:hover{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .td-post-header{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .td-post-title{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--td-text);
            margin-bottom: 8px;
        }
        .td-post-meta{
            font-size: 0.8rem;
            color: var(--td-muted);
        }
        .td-sentiment-badge{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .td-sentiment-badge.positive{ background: #d1fae5; color: #065f46; }
        .td-sentiment-badge.neutral{ background: #e5e7eb; color: #374151; }
        .td-sentiment-badge.negative{ background: #fee2e2; color: #991b1b; }
        .td-link-btn{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 6px 12px;
            background: rgba(124,58,237,0.1);
            color: var(--td-accent);
            border-radius: 10px;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .td-link-btn:hover{
            background: var(--td-accent);
            color: white;
        }

        /* Cluster Cards */
        .td-cluster-card{
            background: var(--td-card);
            border: 1px solid var(--td-border);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        }
        .td-cluster-card:hover{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .td-cluster-label{
            display: inline-block;
            padding: 4px 10px;
            background: linear-gradient(135deg, var(--td-accent) 0%, var(--td-accent-2) 100%);
            color: white;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        /* Skeleton Loaders */
        @keyframes shimmer{
            0%{ background-position: -200% 0; }
            100%{ background-position: 200% 0; }
        }
        .td-skel{
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 8px;
        }
        .td-skel-h{ height: 20px; width: 60%; margin-bottom: 10px; }
        .td-skel-p{ height: 80px; width: 100%; }
        .td-skel-chip{ height: 32px; width: 80px; display: inline-block; margin: 4px; }

        /* Reduced Motion */
        @media (prefers-reduced-motion: reduce){
            *, *::before, *::after{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* Focus Rings */
        a:focus, button:focus{
            outline: 2px solid var(--td-ring);
            outline-offset: 2px;
        }
    </style>
    
    <!-- This is your exact navbar HTML, unchanged -->
    <nav class="td-navbar">
        <div class="td-brand-area">
            <span class="td-logo">🔍</span>
            <span class="td-brand">Trenddit</span>
        </div>
        <div class="td-nav-links">
            <a href="#overview" class="td-nav-link"><span class="td-emoji">🏠</span><span>Overview</span></a>
            <a href="#sentiment-timeline" class="td-nav-link"><span class="td-emoji">📈</span><span>Sentiment</span></a>
            <a href="#top-ngrams" class="td-nav-link"><span class="td-emoji">🧩</span><span>Keywords</span></a>
            <a href="#topic-clusters" class="td-nav-link"><span class="td-emoji">🗂️</span><span>Clusters</span></a>
            <a href="#live-posts" class="td-nav-link"><span class="td-emoji">⚡</span><span>Live</span></a>
            <a href="#alerts" class="td-nav-link"><span class="td-emoji">🚨</span><span>Alerts</span></a>
        </div>
    </nav>
""",
    unsafe_allow_html=True,
)

# ===========================
# AUTHENTICATION UI
# ===========================
if not check_authentication():
    st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔍</div>
            <h1 style="background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent;
                       background-clip: text;
                       font-size: 3rem; font-weight: 800; margin: 0;">
                Trenddit
            </h1>
            <p style="color: #64748b; font-size: 1.1rem; margin-top: 0.5rem;">
                Real-time social media analytics powered by AI
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Initialize session state for auth mode
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "signin"

        # Auth mode toggle buttons
        col_signin, col_signup = st.columns(2)
        with col_signin:
            if st.button(
                "🔑 Sign In",
                use_container_width=True,
                type=(
                    "primary" if st.session_state.auth_mode == "signin" else "secondary"
                ),
            ):
                st.session_state.auth_mode = "signin"
        with col_signup:
            if st.button(
                "✨ Create Account",
                use_container_width=True,
                type=(
                    "primary" if st.session_state.auth_mode == "signup" else "secondary"
                ),
            ):
                st.session_state.auth_mode = "signup"

        st.markdown("<br>", unsafe_allow_html=True)

        # Show appropriate form based on mode
        if st.session_state.auth_mode == "signin":
            st.markdown("### Welcome back!")
            with st.form("signin_form"):
                email = st.text_input(
                    "Email", placeholder="your.email@example.com", key="signin_email"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••",
                    key="signin_password",
                )
                submit = st.form_submit_button("Sign In", use_container_width=True)

                if submit:
                    if email and password:
                        with st.spinner("Signing in..."):
                            result = sign_in(email, password)
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            elif hasattr(result, "user") and result.user:
                                # Successfully signed in
                                st.session_state.authenticated = True
                                st.session_state.user = result.user
                                st.success("✅ Signed in successfully!")
                                time.sleep(0.5)  # Brief pause to show success message
                                st.rerun()
                            else:
                                st.error("❌ Sign in failed. Please try again.")
                    else:
                        st.warning("Please enter both email and password")

        else:  # signup mode
            st.markdown("### Create your account")
            with st.form("signup_form"):
                email = st.text_input(
                    "Email", placeholder="your.email@example.com", key="signup_email"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="••••••••",
                    key="signup_password",
                )
                password_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="••••••••",
                    key="signup_password_confirm",
                )
                submit = st.form_submit_button(
                    "Create Account", use_container_width=True
                )

                if submit:
                    if email and password and password_confirm:
                        if password != password_confirm:
                            st.error("❌ Passwords don't match")
                        elif len(password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        else:
                            with st.spinner("Creating account..."):
                                result = sign_up(email, password)
                                if "error" in result:
                                    st.error(f"❌ {result['error']}")
                                else:
                                    st.success(
                                        "✅ Account created! Please check your email to verify."
                                    )
                                    st.info(
                                        "💡 You can now sign in with your credentials."
                                    )
                    else:
                        st.warning("Please fill in all fields")

    st.stop()

# ===========================
# AUTHENTICATED USER UI
# ===========================

# Add logout button to sidebar
with st.sidebar:
    st.markdown(
        f"### 👤 {st.session_state.user.email if st.session_state.user else 'User'}"
    )
    if st.button("🚪 Sign Out", use_container_width=True):
        if sign_out():
            st.session_state.authenticated = False
            st.session_state.user = None
            st.success("Signed out successfully!")
            st.rerun()

st.markdown('<a id="overview"></a>', unsafe_allow_html=True)
st.title("Trenddit")
st.caption("Real-time insights from across social media communities")

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

# ---- KPI Cards Row ----
if not df.empty:
    total_posts = len(df)
    avg_sentiment = (
        df["sentiment_score"].mean() if "sentiment_score" in df.columns else 0.0
    )

    # Volume change (optional)
    volume_change_str = "—"
    try:
        # Compute a quick timeline to get volume change
        df_temp = df.copy()
        df_temp["sentiment_score"] = pd.to_numeric(
            df_temp["sentiment_score"], errors="coerce"
        ).fillna(0)
        df_temp = df_temp.set_index("created_at").sort_index()
        if hours <= 24:
            vol_series = df_temp["sentiment_score"].resample("15min").count()
        else:
            vol_series = df_temp["sentiment_score"].resample("1H").count()
        if len(vol_series) >= 2:
            last_vol = vol_series.iloc[-1]
            prev_vol = vol_series.iloc[-2]
            if prev_vol > 0:
                change_pct = ((last_vol - prev_vol) / prev_vol) * 100
                volume_change_str = (
                    f"+{change_pct:.1f}%" if change_pct >= 0 else f"{change_pct:.1f}%"
                )
    except:
        pass

    # Top subreddit
    top_subreddit = "—"
    try:
        if "metadata" in df.columns:
            import json

            subreddits = []
            for meta in df["metadata"].dropna():
                if isinstance(meta, str):
                    try:
                        m = json.loads(meta)
                        if "subreddit" in m:
                            subreddits.append(m["subreddit"])
                    except:
                        pass
                elif isinstance(meta, dict) and "subreddit" in meta:
                    subreddits.append(meta["subreddit"])
            if subreddits:
                from collections import Counter

                top_subreddit = f"r/{Counter(subreddits).most_common(1)[0][0]}"
    except:
        pass

    st.markdown(
        f"""
        <div class="td-kpi-row">
            <div class="td-kpi-card">
                <div class="td-kpi-label"><span class="td-kpi-icon">💬</span> Total Posts</div>
                <div class="td-kpi-value">{total_posts:,}</div>
            </div>
            <div class="td-kpi-card">
                <div class="td-kpi-label"><span class="td-kpi-icon">😊</span> Avg Sentiment</div>
                <div class="td-kpi-value">{avg_sentiment:.2f}</div>
            </div>
            <div class="td-kpi-card">
                <div class="td-kpi-label"><span class="td-kpi-icon">📊</span> Volume Change</div>
                <div class="td-kpi-value">{volume_change_str}</div>
            </div>
            <div class="td-kpi-card">
                <div class="td-kpi-label"><span class="td-kpi-icon">#️⃣</span> Top Subreddit</div>
                <div class="td-kpi-value" style="font-size:1.4rem;">{top_subreddit}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Skeleton loaders when empty
    st.markdown(
        """
        <div class="td-kpi-row">
            <div class="td-kpi-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            <div class="td-kpi-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            <div class="td-kpi-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            <div class="td-kpi-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Show matched posts count
if not df.empty:
    st.caption(
        f"📊 {len(df):,} posts matched since {start_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )
else:
    st.caption("📊 No posts matched. Try a different keyword or expand the timeframe.")

# ---- Layout: Left: Charts, Right: Stream -->
left, right = st.columns([2, 1])

with left:
    st.markdown('<a id="sentiment-timeline"></a>', unsafe_allow_html=True)
    # Sentiment timeline: compute hourly average and volume
    st.subheader("Sentiment timeline")
    if df.empty:
        st.info("No posts found for this query/timeframe.")
        # Quick expand button when zero results

        if timeframe != "7d":
            if st.button("🔍 Expand to 7 days", key="expand_7d"):
                st.session_state["expand_clicked"] = True
                st.rerun()
        # Handle the expand action
        if st.session_state.get("expand_clicked"):
            # Set timeframe to 7d by updating the index
            st.session_state["expand_clicked"] = False
            st.info("Tip: Change timeframe selector to '7d' above to see more results.")
    else:
        df = df.sort_values("created_at")
        # ensure numeric
        df["sentiment_score"] = pd.to_numeric(
            df["sentiment_score"], errors="coerce"
        ).fillna(0)
        df.set_index("created_at", inplace=True)
        # resample depending on timeframe
        if hours <= 24:
            res = df["sentiment_score"].resample("15min").mean().ffill()
            vol = df["sentiment_score"].resample("15min").count()
        else:
            res = df["sentiment_score"].resample("1h").mean().ffill()
            vol = df["sentiment_score"].resample("1h").count()
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

        # Render as chips
        chips_html = '<div style="margin-top: 14px;">'
        for _, row in freq_df.head(25).iterrows():
            chips_html += f'<span class="td-chip">{row["term"]} <span class="td-chip-count">{int(row["count"])}</span></span>'
        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="td-skel td-skel-chip"></div><div class="td-skel td-skel-chip"></div><div class="td-skel td-skel-chip"></div>',
            unsafe_allow_html=True,
        )

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
                        # Clean and truncate representative text to avoid raw HTML
                        rep_text_raw = rep.get("title") or rep.get("body", "")
                        rep_text_clean = strip_html_tags(rep_text_raw)
                        rep_text_trunc = (
                            rep_text_clean[:200]
                            if len(rep_text_clean) > 200
                            else rep_text_clean
                        )
                        cluster_summary.append(
                            (
                                c,
                                len(subset),
                                rep_text_trunc,
                                rep.get("url"),
                            )
                        )

                # Render as styled cards
                for c, count, rep_text, url in cluster_summary:
                    import html

                    # Escape HTML in representative text to prevent rendering HTML tags
                    rep_text_escaped = html.escape(rep_text)

                    subreddit_label = "—"
                    try:
                        meta = (
                            df_reset[df_reset["cluster"] == c].iloc[0].get("metadata")
                        )
                        if isinstance(meta, str):
                            import json

                            meta = json.loads(meta)
                        if isinstance(meta, dict) and "subreddit" in meta:
                            subreddit_label = html.escape(f"r/{meta['subreddit']}")
                    except:
                        pass

                    card_html = f"""
                    <div class="td-cluster-card">
                        <div class="td-cluster-label">Cluster {c}: {subreddit_label}</div>
                        <div style="color: var(--td-text); margin-bottom: 10px;">{rep_text_escaped}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: var(--td-muted); font-size: 0.85rem;">{count} posts</span>
                            <a href="{html.escape(url or '#', quote=True)}" target="_blank" class="td-link-btn">View Article ↗</a>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

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
        # Skeleton loaders
        st.markdown(
            """
            <div class="td-post-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            <div class="td-post-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            <div class="td-post-card"><div class="td-skel td-skel-h"></div><div class="td-skel td-skel-p"></div></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for p in posts_page:
            import html

            created = pretty_time_ago(p.get("created_at"))
            score = p.get("sentiment_score") or 0
            label = p.get("sentiment_label") or "neutral"

            # Clean, truncate, and escape title/body to prevent raw HTML rendering
            title_raw = p.get("title") or p.get("body", "")
            title_clean = strip_html_tags(title_raw)
            title_trunc = title_clean[:80] if len(title_clean) > 80 else title_clean
            title = html.escape(title_trunc)

            body_raw = p.get("body") or ""
            body_clean = strip_html_tags(body_raw)
            # Truncate BEFORE escaping to avoid breaking HTML entities
            body_truncated = body_clean[:150] if len(body_clean) > 150 else body_clean
            body = html.escape(body_truncated)
            body_suffix = "..." if len(body_raw) > 150 else ""

            url = p.get("url") or "#"
            source_name = p.get("source", "reddit")

            # Escape author name as well
            author = html.escape(p.get("author") or "Anonymous")

            # Get subreddit from metadata
            subreddit = "—"
            try:
                meta = p.get("metadata")
                if isinstance(meta, str):
                    import json

                    meta = json.loads(meta)
                if isinstance(meta, dict) and "subreddit" in meta:
                    subreddit = html.escape(f"r/{meta['subreddit']}")
            except:
                pass

            # Sentiment emoji and badge class
            sentiment_emoji = "😐"
            sentiment_class = "neutral"
            if label == "positive" or score > 0.05:
                sentiment_emoji = "😃"
                sentiment_class = "positive"
            elif label == "negative" or score < -0.05:
                sentiment_emoji = "😞"
                sentiment_class = "negative"

            post_html = f"""
            <div class="td-post-card">
                <div class="td-post-header">
                    <span class="td-chip">{subreddit}</span>
                    <span class="td-sentiment-badge {sentiment_class}">{sentiment_emoji} {label.title()}</span>
                </div>
                <div class="td-post-title">{title}</div>
                <div class="td-post-meta">
                    {source_name} • {author} • {created} • ⬆ {p.get("score", 0)}
                </div>
                <div style="margin-top: 10px; color: var(--td-muted); font-size: 0.9rem;">
                    {body}{body_suffix}
                </div>
                <div style="margin-top: 12px; text-align: right;">
                    <a href="{html.escape(url, quote=True)}" target="_blank" class="td-link-btn">View on Reddit ↗</a>
                </div>
            </div>
            """
            st.markdown(post_html, unsafe_allow_html=True)

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

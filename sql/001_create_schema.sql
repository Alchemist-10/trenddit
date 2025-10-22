-- 001_create_schema.sql

-- Enable extensions (pg_trgm is helpful for similarity / indexes)
create extension if not exists "pg_trgm";
-- pgvector extension (recommended). If not available, use float8[] fallback below.
create extension if not exists vector;

-- USERS
create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text unique,
  full_name text,
  created_at timestamptz default now()
);

-- QUERIES (saved queries for users)
create table if not exists queries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id) on delete set null,
  keyword text not null,
  sources text[] default array['reddit','twitter'],
  created_at timestamptz default now()
);

-- POSTS
-- Use pgvector vector(384) if using all-MiniLM (384 dims).
create table if not exists posts (
  id text primary key, -- e.g. reddit t3_xxx or twitter tweet id
  source text not null, -- 'reddit' or 'twitter'
  source_id text not null, -- original id
  keyword text, -- query keyword that matched
  title text,
  body text,
  author text,
  url text,
  score integer,
  created_at timestamptz,
  inserted_at timestamptz default now(),
  -- embedding vector for semantic search (384 dims for miniLM)
  embedding vector(384),
  sentiment_score double precision,
  sentiment_label text,
  metadata jsonb
);

-- Indexes
create index if not exists posts_created_at_idx on posts (created_at);
create index if not exists posts_keyword_idx on posts (keyword text_pattern_ops);
create index if not exists posts_source_idx on posts (source);
-- GiST/GIN indexes for pg_trgm on title/body for quick text search
create index if not exists posts_title_trgm on posts using gin (title gin_trgm_ops);
create index if not exists posts_body_trgm on posts using gin (body gin_trgm_ops);

-- AGGREGATES (timeseries aggregates, computed by scheduled job / edge function)
create table if not exists aggregates (
  id uuid primary key default gen_random_uuid(),
  keyword text,
  source text,
  period_start timestamptz,
  period_end timestamptz,
  avg_sentiment double precision,
  volume integer,
  created_at timestamptz default now(),
  details jsonb
);
create index if not exists aggregates_keyword_idx on aggregates (keyword);
create index if not exists aggregates_period_idx on aggregates (period_start);

-- ALERTS
create table if not exists alerts (
  id uuid primary key default gen_random_uuid(),
  query_id uuid references queries(id) on delete set null,
  keyword text,
  source text,
  alert_type text, -- e.g., 'surge', 'negative_spike'
  message text,
  triggered_at timestamptz default now(),
  payload jsonb
);
create index if not exists alerts_keyword_idx on alerts (keyword);

-- EMBEDDINGS (optional separate table if you prefer)
create table if not exists embeddings (
  post_id text primary key references posts(id) on delete cascade,
  vector vector(384)
);

# 🔐 Supabase Authentication Setup Guide

## Prerequisites

Your Streamlit app now has authentication! Follow these steps to enable it in Supabase.

---

## 📋 Supabase Dashboard Setup

### 1. Enable Email Authentication

1. Go to your Supabase Dashboard: https://supabase.com/dashboard
2. Select your project
3. Navigate to **Authentication** → **Providers**
4. Enable **Email** provider (should be enabled by default)
5. Configure email templates (optional):
   - Go to **Authentication** → **Email Templates**
   - Customize "Confirm Signup", "Magic Link", "Reset Password" emails

### 2. Configure Auth Settings

1. Go to **Authentication** → **Settings**
2. **Site URL**: Set to `http://localhost:8501` (for development)
3. **Redirect URLs**: Add `http://localhost:8501`
4. **Email Confirmation**:
   - ✅ Enable "Enable email confirmations" (recommended for production)
   - ⚠️ Disable for local testing if you don't want to verify emails
5. Save changes

### 3. Set Up Row Level Security (RLS) - Optional but Recommended

If you want users to only see their own data:

```sql
-- Enable RLS on posts table
ALTER TABLE posts ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read all posts
CREATE POLICY "Public posts are viewable by everyone"
ON posts FOR SELECT
USING (true);

-- Policy: Only authenticated users can insert posts
CREATE POLICY "Authenticated users can create posts"
ON posts FOR INSERT
WITH CHECK (auth.uid() IS NOT NULL);

-- Enable RLS on other tables (queries, alerts, embeddings)
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read/write their own queries
CREATE POLICY "Users can manage their queries"
ON queries
USING (auth.uid()::text = user_id);
```

---

## 🚀 How to Use

### Sign Up

1. Run `streamlit run app\streamlit_app.py`
2. Click **"✨ Sign Up"** tab
3. Enter email and password (min 6 characters)
4. Click **"Create Account"**
5. _(If email confirmation is enabled)_ Check your email and verify
6. Sign in with your credentials

### Sign In

1. Click **"🔑 Sign In"** tab
2. Enter your email and password
3. Click **"Sign In"**
4. You'll be redirected to the main dashboard

### Sign Out

- Click the **"🚪 Sign Out"** button in the sidebar

---

## 🧪 Testing Without Email Verification

For local development, disable email confirmation:

1. Go to **Authentication** → **Settings**
2. Under **Email Auth**, disable "Enable email confirmations"
3. Users can now sign up and sign in immediately without email verification

---

## 🔑 Features Implemented

✅ **Sign Up** - Create new accounts with email/password  
✅ **Sign In** - Authenticate existing users  
✅ **Sign Out** - Clear session and logout  
✅ **Session Management** - Persistent login state  
✅ **Protected Routes** - Dashboard only accessible when authenticated  
✅ **User Info Display** - Shows logged-in user email in sidebar  
✅ **Modern UI** - Beautiful gradient login forms with tabs

---

## 🛠️ Advanced Configuration

### Add Social Login (Google, GitHub, etc.)

1. Go to **Authentication** → **Providers**
2. Enable desired provider (e.g., Google)
3. Add OAuth credentials from provider
4. Update your Streamlit app to add social login buttons:

```python
# Add to your sign-in UI
if st.button("Sign in with Google"):
    result = supabase.auth.sign_in_with_oauth({
        "provider": "google"
    })
```

### Password Reset Flow

```python
def reset_password(email: str):
    try:
        supabase.auth.reset_password_for_email(email)
        return True
    except Exception as e:
        return {"error": str(e)}
```

---

## 📚 Resources

- [Supabase Auth Docs](https://supabase.com/docs/guides/auth)
- [Supabase Python Client](https://supabase.com/docs/reference/python/introduction)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)

---

## 🐛 Troubleshooting

### "Invalid login credentials"

- Check email and password are correct
- Verify email confirmation is complete (if enabled)
- Check Supabase logs in Dashboard → Logs → Auth Logs

### "User already registered"

- Email already exists in system
- Use sign-in instead of sign-up
- Or use password reset to recover account

### Session not persisting

- Check Supabase connection in `.env`
- Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
- Check browser cookies are enabled

---

Enjoy your authenticated Trenddit app! 🎉

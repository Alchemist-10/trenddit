# app/utils.py
from datetime import datetime, timezone
from dateutil import parser, relativedelta


def pretty_time_ago(ts):
    if not ts:
        return ""
    if isinstance(ts, str):
        try:
            dt = parser.isoparse(ts)
        except:
            return ts
    else:
        dt = ts
    now = datetime.now(tz=dt.tzinfo) if dt.tzinfo else datetime.utcnow()
    diff = now - dt
    seconds = diff.total_seconds()
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds//60)}m ago"
    if seconds < 86400:
        return f"{int(seconds//3600)}h ago"
    return dt.strftime("%Y-%m-%d %H:%M UTC")

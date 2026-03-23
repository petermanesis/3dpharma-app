"""
Passkey-based access control for the 3DPharma app.

Passkey format (before encoding):  <8-char-random-id>|<duration>
  duration = "L" for lifetime, or a number string (hours, e.g. "2", "24")

The raw string is base62-encoded → short, URL-safe, opaque passkey (~12 chars).

First-use timestamp is stored in passkey_store.json.
"""

import json
import os
import random
import string
import streamlit as st
from datetime import datetime, timezone

STORE_FILE = os.path.join(os.path.dirname(__file__), "passkey_store.json")

_B62 = string.ascii_uppercase + string.ascii_lowercase + string.digits  # 62 chars


def _b62_encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    if n == 0:
        return _B62[0]
    result = []
    while n:
        result.append(_B62[n % 62])
        n //= 62
    return "".join(reversed(result))


def _b62_decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 62 + _B62.index(ch)
    length = (n.bit_length() + 7) // 8 or 1
    return n.to_bytes(length, "big")


def _encode(raw: str) -> str:
    return _b62_encode(raw.encode())


def _decode(token: str) -> str | None:
    try:
        return _b62_decode(token).decode()
    except Exception:
        return None


# ── Store helpers ─────────────────────────────────────────────────────────────

def _load_store() -> dict:
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_store(store: dict):
    with open(STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_passkey(duration_hours: float | None = None) -> str:
    """
    Create a short passkey (~12 chars).
    duration_hours=None  → lifetime
    duration_hours=2     → valid 2 hours from first use
    """
    rand_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    duration_tag = "L" if duration_hours is None else str(int(duration_hours))
    raw = f"{rand_id}|{duration_tag}"
    return _encode(raw)


def _validate_passkey(token: str) -> tuple[bool, str]:
    raw = _decode(token)
    if not raw or "|" not in raw:
        return False, "Invalid passkey."

    rand_id, duration_tag = raw.split("|", 1)

    duration_hours: float | None
    if duration_tag == "L":
        duration_hours = None
    else:
        try:
            duration_hours = float(duration_tag)
        except ValueError:
            return False, "Invalid passkey."

    store = _load_store()

    if rand_id not in store:
        store[rand_id] = {"first_used": datetime.now(timezone.utc).isoformat()}
        _save_store(store)
        return True, "Access granted."

    if duration_hours is None:
        return True, "Access granted (lifetime key)."

    first_used = datetime.fromisoformat(store[rand_id]["first_used"])
    elapsed_hours = (datetime.now(timezone.utc) - first_used).total_seconds() / 3600

    if elapsed_hours <= duration_hours:
        return True, "Access granted."

    return False, (
        f"Passkey expired. It was valid for {int(duration_hours)}h from first use "
        f"({first_used.strftime('%Y-%m-%d %H:%M UTC')})."
    )


def check_auth():
    """
    Call at the top of main(). Blocks with st.stop() until a valid passkey is entered.
    """
    if st.session_state.get("_authenticated"):
        return

    st.set_page_config(page_title="Access Required", page_icon="🔐")
    st.title("🔐 Access Required")
    st.markdown("Please enter your passkey to access this application.")

    token = st.text_input("Passkey", type="password", key="_passkey_input")

    if st.button("Submit", type="primary"):
        if not token.strip():
            st.error("Please enter a passkey.")
        else:
            valid, msg = _validate_passkey(token.strip())
            if valid:
                st.session_state["_authenticated"] = True
                st.rerun()
            else:
                st.error(msg)

    st.stop()

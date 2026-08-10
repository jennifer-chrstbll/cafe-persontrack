import time
import logging
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger("supabase_client")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Supabase Direct Write Client
# Replaces the old FastAPI HTTP backend client.
# Writes occupancy data directly to Supabase using the supabase-py SDK.
# ---------------------------------------------------------------------------

_supabase_client = None

def _get_supabase():
    """Lazy-initialise Supabase client (singleton)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.warning("[SupabaseClient] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — occupancy sync DISABLED.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        logger.info("[SupabaseClient] Connected to Supabase.")
        return _supabase_client
    except Exception as e:
        logger.error(f"[SupabaseClient] Failed to connect: {e}")
        return None


def push_occupancy(camera_id: str, floor: int, person_count: int) -> bool:
    """
    Upserts the current person count for a given camera/floor into the
    `occupancy` table in Supabase.

    Table schema (run once in Supabase SQL Editor):
        CREATE TABLE IF NOT EXISTS occupancy (
            camera_id   TEXT PRIMARY KEY,
            floor       INTEGER NOT NULL,
            person_count INTEGER NOT NULL DEFAULT 0,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        );

    The dashboard reads this table (or polls the API endpoint) to show
    real-time floor occupancy.
    """
    sb = _get_supabase()
    if sb is None:
        return False
    try:
        sb.table("occupancy").upsert({
            "camera_id": camera_id,
            "floor": floor,
            "person_count": person_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, on_conflict="camera_id").execute()
        return True
    except Exception as e:
        logger.debug(f"[SupabaseClient] push_occupancy error: {e}")
        return False


import threading

def push_occupancy_background(camera_id: str, floor: int, person_count: int):
    """Non-blocking fire-and-forget occupancy sync — won't block the video loop."""
    t = threading.Thread(target=push_occupancy, args=(camera_id, floor, person_count), daemon=True)
    t.start()

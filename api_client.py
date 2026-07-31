import time
import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger("api_client")
logging.basicConfig(level=logging.INFO)

class CRMBackendClient:
    """
    Asynchronous API Client for sending Person Tracking & Occupancy telemetry to cafe-crm backend.
    Runs non-blocking updates in background worker threads.
    """
    def __init__(self, backend_url: str = config.BACKEND_API_URL):
        self.backend_url = backend_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=3.0)

    async def send_track_updates(self, camera_id: str, tracks_data: List[Dict[str, Any]]) -> bool:
        """
        Sends list of active tracks to backend.
        Payload structure:
        [
            {
                "camera_id": "CAM_1",
                "raw_track_id": "GT-0001",
                "pos_x": 450.5,
                "pos_y": 320.0,
                "velocity_x": 1.2,
                "velocity_y": -0.5,
                "status": "ACTIVE"
            }, ...
        ]
        """
        if not tracks_data:
            return True

        url = f"{self.backend_url}/tracks/update"
        payload = {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "tracks": tracks_data
        }
        try:
            resp = await self.client.post(url, json=payload)
            return resp.status_code in (200, 201)
        except Exception as e:
            # Silent fallback if CRM backend is offline
            logger.debug(f"[CRMBackendClient] Track update sync error: {e}")
            return False

    async def send_occupancy_log(self, camera_id: str, floor: int, person_count: int) -> bool:
        """
        Sends occupancy count telemetry to backend.
        """
        url = f"{self.backend_url}/analytics/occupancy"
        payload = {
            "camera_id": camera_id,
            "floor": floor,
            "person_count": person_count,
            "timestamp": time.time()
        }
        try:
            resp = await self.client.post(url, json=payload)
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"[CRMBackendClient] Occupancy log sync error: {e}")
            return False

    async def bind_identity(self, customer_id: str, camera_id: str = "CAM_1", pos_x: float = 640.0, pos_y: float = 360.0) -> Dict[str, Any]:
        """
        Triggers Identity Association Engine at backend (/association/bind)
        when a face recognition event occurs at cashier POS.
        """
        url = f"{self.backend_url}/association/bind"
        payload = {
            "customer_id": customer_id,
            "camera_id": camera_id,
            "pos_x": pos_x,
            "pos_y": pos_y
        }
        try:
            resp = await self.client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "UNMATCHED", "reason": f"HTTP_{resp.status_code}"}
        except Exception as e:
            logger.debug(f"[CRMBackendClient] Identity bind error: {e}")
            return {"status": "UNMATCHED", "reason": str(e)}

    async def close(self):
        await self.client.aclose()


import threading

# Helper to run async sync function in background thread without blocking main video loop
def sync_telemetry_background(client: CRMBackendClient, camera_id: str, floor: int, tracks_data: List[Dict[str, Any]]):
    def _worker():
        async def _async_call():
            await client.send_track_updates(camera_id, tracks_data)
            await client.send_occupancy_log(camera_id, floor, len(tracks_data))
        try:
            asyncio.run(_async_call())
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

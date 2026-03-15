from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from detector import RobotDetector, VideoStream
from events import SCOUTING_FIELDS, empty_scouting_payload
from storage import create_match, end_match as close_match, get_match_summary, init_storage, insert_event


class StreamTracker:
    def __init__(self) -> None:
        init_storage()
        self.detector = RobotDetector()
        self.stream = VideoStream()

        self.active_match: Optional[Dict[str, Any]] = None
        self.stream_running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[Dict[str, Any]] = None

    def start_match(self, match_number: int, team_number: int) -> int:
        with self._lock:
            if self.active_match:
                raise RuntimeError("A match is already active")
            match_id = create_match(match_number=match_number, team_number=team_number)
            self.active_match = {
                "id": match_id,
                "match_number": match_number,
                "team_number": team_number,
            }
            return match_id

    def end_match(self) -> Dict[str, Any]:
        with self._lock:
            if not self.active_match:
                raise RuntimeError("No active match")
            match_id = self.active_match["id"]
            close_match(match_id)
            summary = get_match_summary(match_id) or {"id": match_id}
            self.active_match = None
            return summary

    def start_stream(self, stream_url: str) -> None:
        with self._lock:
            if self.stream_running:
                raise RuntimeError("Stream already running")
            self.stream.open(stream_url)
            self.stream_running = True
            self._stream_thread = threading.Thread(target=self._loop, daemon=True)
            self._stream_thread.start()

    def stop_stream(self) -> None:
        with self._lock:
            self.stream_running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
            self._stream_thread = None
        self.stream.close()

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_snapshot

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self.stream_running:
                    break

            ok, frame = self.stream.read()
            if not ok:
                time.sleep(0.2)
                continue

            detections = self.detector.detect(frame)
            snapshot = self._estimate_fields(detections)

            with self._lock:
                self._latest_snapshot = snapshot
                active_match = self.active_match

            if active_match:
                insert_event(active_match["id"], snapshot)

            time.sleep(0.2)

    def _estimate_fields(self, detections) -> Dict[str, Dict[str, float | int | str]]:
        payload = empty_scouting_payload()
        n = len(detections)
        mean_conf = sum(d.confidence for d in detections) / n if n > 0 else 0.0

        # Simple baseline heuristics for autonomous local scouting estimation.
        payload["cycling"] = {"value": int(n > 2), "confidence": min(1.0, 0.4 + mean_conf)}
        payload["scoring"] = {"value": int(n > 0), "confidence": mean_conf}
        payload["feeding"] = {"value": int(n >= 2), "confidence": min(1.0, mean_conf * 0.9)}
        payload["defending"] = {"value": int(n >= 3), "confidence": min(1.0, mean_conf * 0.85)}
        payload["immobile"] = {"value": int(n == 0), "confidence": 0.8 if n == 0 else 0.2}
        payload["intake_type_ground"] = {"value": int(n > 1), "confidence": min(1.0, mean_conf * 0.7)}
        payload["intake_type_outpost_or_source"] = {"value": int(n > 0), "confidence": min(1.0, mean_conf * 0.6)}
        payload["traversal_trench"] = {"value": int(n > 2), "confidence": min(1.0, mean_conf * 0.5)}
        payload["traversal_bump"] = {"value": int(n > 1), "confidence": min(1.0, mean_conf * 0.5)}
        payload["auto_climb_status"] = {
            "value": "engaged" if n > 0 else "unknown",
            "confidence": min(1.0, 0.3 + mean_conf),
        }
        payload["beached_on_fuel"] = {"value": int(n == 0), "confidence": 0.6 if n == 0 else 0.2}
        payload["beached_on_bump"] = {"value": int(n == 0), "confidence": 0.6 if n == 0 else 0.2}
        payload["scores_while_moving"] = {"value": int(n > 1), "confidence": min(1.0, mean_conf * 0.9)}
        payload["robot_broke"] = {"value": int(n == 0), "confidence": 0.75 if n == 0 else 0.15}
        payload["estimated_points_scored"] = {
            "value": int(round(mean_conf * 10 + n * 2)),
            "confidence": min(1.0, mean_conf),
        }

        for field in SCOUTING_FIELDS:
            if "confidence" not in payload[field]:
                payload[field]["confidence"] = 0.0
        return payload

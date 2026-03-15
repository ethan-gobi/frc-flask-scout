from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from detector import RobotDetector, VideoStream
from events import SCOUTING_FIELDS, empty_scouting_payload
from storage import (
    create_match,
    end_match as close_match,
    get_match_summary,
    init_storage,
    insert_event,
    mark_match_started,
    update_match_source,
)


class StreamTracker:
    def __init__(self) -> None:
        init_storage()
        self.logger = logging.getLogger(__name__)
        self.detector = RobotDetector()
        self.stream = VideoStream()

        # Runtime-only state must always start clean on app boot.
        self.active_match: Optional[Dict[str, Any]] = None
        self.stream_running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self.source_input: Optional[str] = None
        self.source_mode: str = "auto"
        self.resolved_source_url: Optional[str] = None
        self.source_type: Optional[str] = None
        self._processing_delay_s = 0.2

    def configure_match(
        self,
        match_number: int,
        source_input: str,
        source_mode: str = "auto",
        team_number: int = 0,
        match_type: str = "qualification",
    ) -> int:
        clean_source = source_input.strip()
        if match_number <= 0:
            raise RuntimeError("Match ID is required")
        if not clean_source:
            raise RuntimeError("Stream URL or video path is required")

        with self._lock:
            if self.stream_running:
                raise RuntimeError("Stop tracking before changing match setup")
            if self.active_match:
                raise RuntimeError("A match is already configured")

            normalized_match_type = (match_type or "qualification").strip().lower()
            match_id = create_match(match_number=match_number, team_number=team_number, match_type=normalized_match_type)
            self.active_match = {
                "id": match_id,
                "match_number": match_number,
                "team_number": team_number,
                "match_type": normalized_match_type,
            }
            self.source_input = clean_source
            self.source_mode = source_mode or "auto"
            self.resolved_source_url = None
            self.source_type = None
            update_match_source(
                match_id=match_id,
                source_input=clean_source,
                source_mode=self.source_mode,
                match_type=normalized_match_type,
            )
            return match_id

    def start_tracking(self) -> str:
        with self._lock:
            if self.stream_running:
                raise RuntimeError("Tracking is already running")
            if not self.active_match:
                raise RuntimeError("Match ID is required")
            if not self.source_input:
                raise RuntimeError("Stream URL or video path is required")

            actual_url = self.resolve_stream_url(self.source_input)
            source_type = self._detect_source_type(self.source_input, self.source_mode)
            self._processing_delay_s = 0.0 if source_type in {"recorded", "local_file"} else 0.2

            self.logger.info(
                "Attempting to open source. source_input=%s resolved_url=%s source_type=%s",
                self.source_input,
                actual_url,
                source_type,
            )

            self.stream.open(actual_url)
            self.logger.info("OpenCV source open success. resolved_url=%s", actual_url)

            mark_match_started(self.active_match["id"])
            self.resolved_source_url = actual_url
            self.source_type = source_type
            self.stream_running = True
            self._stream_thread = threading.Thread(target=self._loop, daemon=True)
            self._stream_thread.start()
            return actual_url

    def stop_stream(self) -> None:
        with self._lock:
            self.stream_running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
            self._stream_thread = None
        self.stream.close()

    def end_match(self) -> Dict[str, Any]:
        with self._lock:
            if not self.active_match:
                raise RuntimeError("No active match")
            match_id = self.active_match["id"]

        self.stop_stream()
        close_match(match_id)
        summary = get_match_summary(match_id) or {"id": match_id}

        with self._lock:
            self.active_match = None
            self.source_input = None
            self.source_mode = "auto"
            self.resolved_source_url = None
            self.source_type = None

        return summary

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_snapshot

    def resolve_stream_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        is_youtube = ("youtube.com" in host and "/watch" in parsed.path) or "youtu.be" in host

        self.logger.info("Source URL received. original_url=%s is_youtube=%s", url, is_youtube)
        if not is_youtube:
            return url

        cmd = [sys.executable, "-m", "yt_dlp", "-g", url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            self.logger.warning("yt-dlp resolution failed. original_url=%s error=%s", url, stderr)
            raise RuntimeError(f"Failed to resolve YouTube stream URL: {stderr or 'yt-dlp returned non-zero exit'}")

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            self.logger.warning("yt-dlp returned no playable URLs. original_url=%s", url)
            raise RuntimeError("Failed to resolve YouTube stream URL: no playable media URL returned")

        resolved = lines[0]
        self.logger.info("YouTube URL resolution succeeded. original_url=%s resolved_url=%s", url, resolved)
        return resolved

    def _detect_source_type(self, source_input: str, source_mode: str) -> str:
        mode = (source_mode or "auto").lower()
        if mode in {"live", "recorded", "local_file"}:
            return mode

        parsed = urlparse(source_input)
        if parsed.scheme in {"", "file"} and Path(source_input).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            return "local_file"

        if "youtube.com" in parsed.netloc.lower() or "youtu.be" in parsed.netloc.lower():
            return "recorded"

        if parsed.scheme in {"rtsp", "rtmp"}:
            return "live"

        return "live"

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            configured = self.active_match is not None
            tracking = self.stream_running
            source_type = self.source_type
        setup_status = "Match configured" if configured else "No match configured"
        tracking_status = "Tracking running" if tracking else "Tracking stopped"
        source_status = "No source loaded"
        if source_type == "recorded":
            source_status = "Recorded source loaded"
        elif source_type == "live":
            source_status = "Live source loaded"
        elif source_type == "local_file":
            source_status = "Recorded source loaded"

        return {
            "setup_status": setup_status,
            "tracking_status": tracking_status,
            "source_status": source_status,
            "source_mode": self.source_mode,
            "source_input": self.source_input,
            "resolved_source_url": self.resolved_source_url,
            "match_type": self.active_match["match_type"] if self.active_match else None,
        }

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self.stream_running:
                    break
                active_match = self.active_match
                delay = self._processing_delay_s

            ok, frame = self.stream.read()
            if not ok:
                with self._lock:
                    self.stream_running = False
                break

            detections = self.detector.detect(frame)
            snapshot = self._estimate_fields(detections)

            with self._lock:
                self._latest_snapshot = snapshot

            if active_match:
                insert_event(active_match["id"], snapshot)

            if delay > 0:
                time.sleep(delay)

    def _estimate_fields(self, detections) -> Dict[str, Dict[str, float | int | str]]:
        payload = empty_scouting_payload()
        n = len(detections)
        mean_conf = sum(d.confidence for d in detections) / n if n > 0 else 0.0

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

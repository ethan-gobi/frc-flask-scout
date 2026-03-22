"""
FRC REBUILT 2026 — Grounded Scout v3 (Claude Code Reference)
=============================================================
This file is a well-structured starting point for Claude Code to
improve upon with its own decision making.

Claude Vision is used for EVERYTHING possible:
  - Bumper number reading (instead of Tesseract alone)
  - Match phase detection
  - Scoreboard OCR
  - Game piece (fuel) detection
  - Robot role inference
  - Defense detection
  - Climbing detection
  - Field position estimation
  - Beaching detection
  - Intake type detection
  - Whether the stream is even showing an FRC match
  - Pre-match robot selection scoring

Tesseract OCR is kept as a FALLBACK only for bumper numbers
when Claude is uncertain, not as the primary method.

Known TODOs for Claude Code to resolve:
  - Main run() loop needs completion
  - Flask integration (convert to tracker.py in Flask app)
  - Reconnection logic for dropped streams
  - Endgame phase awareness in state machine
  - Cost tracking and estimation
  - Replace print() with Python logging module
  - Unit tests for FrameAnalysis gating
  - Google Vision provider implementation

Design principles to preserve:
  - Null > guess. Always.
  - Non-FRC video → all nulls + error_flag, never crashes
  - Every Claude output is confidence-gated before use
  - Vision providers are swappable (Claude now, Google Vision later)
  - TBA is ground truth for scores — vision is secondary
  - Claude is primary identity tool — Tesseract is fallback only

Requirements:
  pip install anthropic opencv-python yt-dlp Pillow
  pip install supervision pytesseract requests ultralytics

  export ANTHROPIC_API_KEY=sk-ant-...
  export TBA_API_KEY=your_tba_key

Usage:
  python frc_scout_v3.py --url "https://twitch.tv/firstinspires" \
                          --event 2026onsc --match 14

  python frc_scout_v3.py --url "./match_14.mp4" \
                          --event 2026onsc --match 14 --alliance red

  # Future: swap provider
  python frc_scout_v3.py --url "..." --event 2026onsc --match 14 \
                          --provider google
"""

from __future__ import annotations

import abc
import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import cv2
import numpy as np
import pytesseract
import requests
import supervision as sv
from ultralytics import YOLO


# ─── Configuration ───────────────────────────────────────────────────
CLAUDE_MODEL     = "claude-sonnet-4-6"
MAX_TOKENS       = 2048
TBA_BASE         = "https://www.thebluealliance.com/api/v3"
TBA_KEY          = os.getenv("TBA_API_KEY", "")
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
YOLO_WEIGHTS     = "yolov8n.pt"
OUTPUT_DIR       = "scouting_data"

# Confidence gates — below these we output null, never guess
MIN_FRC_CONF     = 0.80   # confidence frame IS an FRC match
MIN_ROBOT_CONF   = 0.75   # confidence in robot-level behavioral fields
MIN_OCR_VOTES    = 0.45   # fraction of pre-match frames that must agree on team number
MIN_BUMPER_CONF  = 0.70   # Claude bumper read confidence threshold

# Bumper OCR (Tesseract fallback only)
UPSCALE          = 3
BUMPER_FRAC      = 0.25   # bottom 25% of robot bbox = bumper zone
PRE_MATCH_S      = 25     # seconds to run identity detection before auto

# Timing
VISION_INTERVAL  = 3.0    # seconds between full Claude vision calls during teleop
PREMATH_INTERVAL = 1.5    # faster during pre-match for better identity coverage
MIN_OBS          = 10     # minimum valid observations before deriving fields


# ════════════════════════════════════════════════════════════════════
# FRAME ANALYSIS — normalized output from any vision provider
# ════════════════════════════════════════════════════════════════════

class FrameAnalysis:
    """
    Normalized, provider-agnostic output from vision analysis.
    All fields default to None. Providers must never guess.
    apply_gates() enforces confidence thresholds after provider returns.
    """

    def __init__(self):
        # ── Match level ──────────────────────────────────────────
        self.is_frc_match:        bool | None = None
        self.frc_confidence:      float       = 0.0
        self.match_phase:         str | None  = None  # pre_match|auto|teleop|endgame|post_match
        self.error_flag:          str | None  = None  # not_frc|low_visibility|provider_error|...

        # ── Scoreboard (read from overlay — most reliable source) ─
        self.red_score:           int | None  = None
        self.blue_score:          int | None  = None
        self.timer:               str | None  = None
        self.red_teams:           list[int]   = []
        self.blue_teams:          list[int]   = []

        # ── Target robot identity ────────────────────────────────
        self.robot_visible:       bool | None = None
        self.robot_confidence:    float       = 0.0
        self.bumper_number:       int | None  = None   # Claude-read bumper number
        self.bumper_number_conf:  float       = 0.0
        self.bumper_alliance:     str | None  = None   # "red"|"blue" from bumper color
        self.robot_grid_position: tuple | None = None  # (row, col) on 6x3 field grid

        # ── Robot behavior (all gated by robot_confidence) ────────
        self.has_fuel:            bool | None = None   # carrying game pieces
        self.near_hub:            bool | None = None   # close to scoring hub
        self.near_opponent_zone:  bool | None = None   # in opponent's half
        self.near_outpost:        bool | None = None   # near human player outpost
        self.is_moving:           bool | None = None
        self.is_climbing:         bool | None = None
        self.is_defending:        bool | None = None   # actively blocking opponent
        self.intake_type:         str | None  = None   # "ground"|"outpost"|None
        self.appears_stuck:       bool | None = None   # possible beaching
        self.stuck_on:            str | None  = None   # "fuel"|"bump"|None
        self.just_scored:         bool | None = None   # fuel entered hub this frame
        self.scoring_location:    str | None  = None   # "high_hub"|"low_hub"|None
        self.robot_action:        str | None  = None   # free text: what robot is doing

        # ── Field context ─────────────────────────────────────────
        self.visible_robots:      int | None  = None   # total robots visible in frame
        self.fuel_in_play:        int | None  = None   # approximate fuel pieces visible
        self.camera_angle:        str | None  = None   # "overhead"|"field_level"|"unknown"

        # ── Token usage (Claude only) ──────────────────────────────
        self._input_tokens:  int = 0
        self._output_tokens: int = 0

    def apply_gates(self) -> "FrameAnalysis":
        """
        Enforce confidence thresholds in-place.
        Any field below its threshold becomes None.
        Call this after every provider returns.
        """
        # Gate 1: not FRC
        if not self.is_frc_match or self.frc_confidence < MIN_FRC_CONF:
            self.is_frc_match  = False
            self.error_flag    = self.error_flag or "not_frc"
            self._null_robot_fields()
            self._null_scoreboard()
            return self

        # Gate 2: robot not visible
        if not self.robot_visible:
            self.error_flag = "low_visibility"
            self._null_robot_fields()
            return self

        # Gate 3: robot confidence too low
        if self.robot_confidence < MIN_ROBOT_CONF:
            self.error_flag = "low_robot_confidence"
            self._null_robot_fields()
            return self

        # Gate 4: bumper number confidence
        if self.bumper_number_conf < MIN_BUMPER_CONF:
            self.bumper_number = None

        return self

    def _null_robot_fields(self):
        self.has_fuel = self.near_hub = self.near_opponent_zone = None
        self.near_outpost = self.is_moving = self.is_climbing = None
        self.is_defending = self.intake_type = self.appears_stuck = None
        self.stuck_on = self.just_scored = self.scoring_location = None
        self.robot_action = self.robot_grid_position = None
        self.robot_confidence = 0.0

    def _null_scoreboard(self):
        self.red_score = self.blue_score = self.timer = None


# ════════════════════════════════════════════════════════════════════
# VISION PROVIDER ABSTRACTION
# ════════════════════════════════════════════════════════════════════

class VisionProvider(abc.ABC):
    """
    Abstract base for all vision providers.

    To add Google Vision:
      1. class GoogleVisionProvider(VisionProvider)
      2. Implement analyze_frame() and identify_robot()
      3. Add to PROVIDERS dict at bottom of file
      4. Pass --provider google at CLI

    Nothing else in the pipeline needs to change.
    """

    @abc.abstractmethod
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        alliance: str,
        context: str,
    ) -> FrameAnalysis:
        """
        Full match analysis — called every VISION_INTERVAL seconds.
        Must never raise. Must never guess. Return None fields when uncertain.
        """
        ...

    @abc.abstractmethod
    def identify_robot(
        self,
        frame: np.ndarray,
        bbox: tuple,
        known_teams: list[int],
    ) -> tuple[int | None, float]:
        """
        Read bumper number from a robot crop.
        Returns (team_number, confidence) or (None, 0.0).
        Called during pre-match window for every tracked robot.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__


# ─── Claude Vision Provider ──────────────────────────────────────────

_CLAUDE_ANALYSIS_PROMPT = """You are a strict FRC (FIRST Robotics Competition) vision analyst \
for the 2026 game REBUILT. You analyze video frames from match livestreams.

━━ ABSOLUTE RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER guess. If you cannot directly observe something, set it to null.
   Null is always correct. A wrong non-null value is never acceptable.

2. If this frame is NOT an FRC match (commercials, speeches, crowd shots,
   other sports, menus, black frames, anything non-FRC) → set is_frc_match
   to false and return immediately with all other fields null.

3. Only report what is EXPLICITLY VISIBLE in THIS frame. No inference
   about what happened before or after. No assumptions about off-screen events.

4. Confidence must be honest. 1.0 = certain beyond doubt.
   0.6 = uncertain. Never inflate. The system will null fields below 0.75.

5. Bumper numbers: read directly from the physical bumper text in the frame.
   If blurry, occluded, wrong angle, or any doubt → bumper_number: null.

6. robot_confidence covers ALL robot behavior fields simultaneously.
   If the robot is even slightly unclear, set it below 0.75.

━━ GAME CONTEXT — REBUILT 2026 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- RED alliance (3 robots) vs BLUE alliance (3 robots)
- FUEL = game pieces (balls). Robots collect and score in the HUB.
- HUB has HIGH GOAL (upper opening) and LOW GOAL (lower slot).
- OUTPOST = human player station where FUEL is delivered to robots.
- TOWER = endgame climbing structure. Multiple levels.
- TRENCH = low passage under a structure robots can drive through.
- BUMP = raised ridge robots drive over.
- Phases: pre_match → auto (15s) → teleop (2m25s) → endgame (last 30s) → post_match
- Endgame starts when ~30s remain on timer.
- Scoreboard overlay is at top of screen — use it for scores and timer.

━━ WHAT TO DETECT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCOREBOARD: Read team numbers, scores, timer from the overlay. This is the
most reliable data source — prioritize it.

TARGET ROBOT: The alliance is given per-frame. Focus on that robot.
For behaviors: observe what the robot is ACTUALLY doing right now.

FIELD POSITION: Use a 6-column x 3-row grid (0-indexed col 0-5, row 0-2).
Row 0 = red alliance wall side. Row 2 = blue alliance wall side.
Col 0 = leftmost when viewing from broadcast camera.

FUEL/GAME PIECES: Balls on the field or in/on a robot.

SCORING EVENTS: Fuel visibly entering the hub opening counts as just_scored.

DEFENSE: Robot is physically blocking, bumping, or shadowing an opponent.
Normal driving near opponents does NOT count as defense.

BEACHING: Robot appears stopped and unable to move (stuck on game piece or bump).

━━ RESPONSE FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Valid JSON only. No markdown. No text outside the JSON.

{
  "is_frc_match": true | false,
  "frc_confidence": 0.0-1.0,
  "match_phase": "pre_match" | "auto" | "teleop" | "endgame" | "post_match" | null,
  "camera_angle": "overhead" | "field_level" | "unknown" | null,
  "scoreboard": {
    "red_score": integer | null,
    "blue_score": integer | null,
    "timer": "2:14" | null,
    "red_teams": [integer, integer, integer] | [],
    "blue_teams": [integer, integer, integer] | []
  },
  "field_context": {
    "visible_robots": integer | null,
    "fuel_in_play": integer | null
  },
  "target_robot_visible": true | false | null,
  "robot_confidence": 0.0-1.0,
  "bumper_number": integer | null,
  "bumper_number_confidence": 0.0-1.0,
  "bumper_alliance": "red" | "blue" | null,
  "robot_grid_position": {"row": 0-2, "col": 0-5} | null,
  "target_robot": {
    "has_fuel": true | false | null,
    "near_hub": true | false | null,
    "near_opponent_zone": true | false | null,
    "near_outpost": true | false | null,
    "is_moving": true | false | null,
    "is_climbing": true | false | null,
    "is_defending": true | false | null,
    "intake_type": "ground" | "outpost" | null,
    "appears_stuck": true | false | null,
    "stuck_on": "fuel" | "bump" | null,
    "just_scored": true | false | null,
    "scoring_location": "high_hub" | "low_hub" | null,
    "robot_action": "<1 sentence description of what robot is doing right now or null>"
  },
  "error_flag": null | "not_frc" | "low_visibility" | "frame_unclear"
}

When is_frc_match is false, return EXACTLY:
{
  "is_frc_match": false,
  "frc_confidence": <confidence it is NOT frc>,
  "match_phase": null,
  "camera_angle": null,
  "scoreboard": {"red_score": null, "blue_score": null, "timer": null,
                 "red_teams": [], "blue_teams": []},
  "field_context": {"visible_robots": null, "fuel_in_play": null},
  "target_robot_visible": null,
  "robot_confidence": 0.0,
  "bumper_number": null,
  "bumper_number_confidence": 0.0,
  "bumper_alliance": null,
  "robot_grid_position": null,
  "target_robot": {
    "has_fuel": null, "near_hub": null, "near_opponent_zone": null,
    "near_outpost": null, "is_moving": null, "is_climbing": null,
    "is_defending": null, "intake_type": null, "appears_stuck": null,
    "stuck_on": null, "just_scored": null, "scoring_location": null,
    "robot_action": null
  },
  "error_flag": "not_frc"
}
"""

_CLAUDE_BUMPER_PROMPT = """You are reading a robot bumper number from a cropped, upscaled image.

The bumper is the padded border around an FRC robot.
Team numbers are printed on the bumper in large digits.
Numbers are between 1 and 9999.

RULES:
- If you can read the number clearly → return it with high confidence.
- If the image is blurry, the number is obscured, or you have ANY doubt → return null.
- Do not guess. Do not infer from partial digits.
- The known possible teams are provided — use them to validate, but do not guess
  if the image doesn't clearly show one of them.

Respond with valid JSON only:
{
  "team_number": integer | null,
  "confidence": 0.0-1.0,
  "alliance_color": "red" | "blue" | null,
  "reasoning": "<one sentence on what you could or could not see>"
}
"""


class ClaudeVisionProvider(VisionProvider):
    """
    Vision provider backed by Claude claude-sonnet-4-6.
    Claude is used for ALL vision tasks:
      - Full frame match analysis (analyze_frame)
      - Bumper number reading (identify_robot)
    Tesseract is kept as a fallback inside identify_robot only.
    """

    def __init__(self):
        if not ANTHROPIC_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        self._total_input_tokens  = 0
        self._total_output_tokens = 0

    def name(self) -> str:
        return f"Claude ({CLAUDE_MODEL})"

    def _encode_frame(self, frame: np.ndarray, quality: int = 78) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.standard_b64encode(buf.tobytes()).decode()

    def _call_claude(
        self,
        system: str,
        user_text: str,
        image_b64: str,
        label: str = "frame",
    ) -> dict | None:
        """
        Single Claude API call with image.
        Returns parsed JSON or None on any failure.
        Never raises.
        """
        try:
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                    ],
                }],
            )
            self._total_input_tokens  += resp.usage.input_tokens
            self._total_output_tokens += resp.usage.output_tokens

            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  [!] Claude JSON parse error ({label}): {e}")
            return None
        except anthropic.APIError as e:
            print(f"  [!] Claude API error ({label}): {e}")
            return None
        except Exception as e:
            print(f"  [!] Unexpected Claude error ({label}): {e}")
            return None

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        alliance: str,
        context: str,
    ) -> FrameAnalysis:
        """
        Full frame analysis — called every VISION_INTERVAL seconds.
        Uses the comprehensive prompt to extract everything Claude can see.
        """
        result = FrameAnalysis()
        result.error_flag = "provider_error"  # default until we succeed

        img_b64 = self._encode_frame(frame)
        user_text = (
            f"Context: {context}\n"
            f"Target alliance: {alliance.upper()}\n"
            f"Frame #{frame_number}. Analyze strictly per your rules."
        )

        raw = self._call_claude(
            system=_CLAUDE_ANALYSIS_PROMPT,
            user_text=user_text,
            image_b64=img_b64,
            label=f"frame-{frame_number}",
        )

        if raw is None:
            return result.apply_gates()

        result = self._map_analysis(raw)
        result._input_tokens  = 0  # tracked at provider level
        result._output_tokens = 0
        return result.apply_gates()

    def identify_robot(
        self,
        frame: np.ndarray,
        bbox: tuple,
        known_teams: list[int],
    ) -> tuple[int | None, float]:
        """
        Primary: Ask Claude to read the bumper number from a cropped robot image.
        Fallback: Tesseract if Claude returns null or low confidence.
        Returns (team_number, confidence).
        """
        # Crop bumper region and upscale 3x
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h    = y2 - y1
        crop = frame[int(y1 + h * (1 - BUMPER_FRAC)):y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        upscaled = cv2.resize(
            crop,
            (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
            interpolation=cv2.INTER_CUBIC,
        )

        # ── Claude primary read ───────────────────────────────────
        img_b64   = self._encode_frame(upscaled, quality=85)
        teams_str = ", ".join(str(t) for t in known_teams) if known_teams else "unknown"
        user_text = (
            f"Known possible teams on this alliance: [{teams_str}]\n"
            f"Read the bumper number from this upscaled robot crop."
        )

        raw = self._call_claude(
            system=_CLAUDE_BUMPER_PROMPT,
            user_text=user_text,
            image_b64=img_b64,
            label="bumper",
        )

        claude_team = None
        claude_conf = 0.0
        if raw:
            claude_team = raw.get("team_number")
            claude_conf = float(raw.get("confidence", 0.0))

        # Validate Claude's read against known TBA teams
        if claude_team and known_teams:
            if claude_team in known_teams:
                pass  # exact match
            else:
                # Check single digit misread
                corrected = _tba_fuzzy_match(claude_team, known_teams)
                if corrected:
                    claude_team = corrected
                else:
                    claude_team = None
                    claude_conf = 0.0

        if claude_team and claude_conf >= MIN_BUMPER_CONF:
            return claude_team, claude_conf

        # ── Tesseract fallback ────────────────────────────────────
        print("  [~] Claude bumper uncertain — trying Tesseract fallback")
        tess_team = _tesseract_read_bumper(upscaled)
        tess_team = _tba_fuzzy_match(tess_team, known_teams) if tess_team else None
        if tess_team:
            return tess_team, 0.60  # Tesseract gets lower base confidence

        return None, 0.0

    def cost_estimate(self) -> float:
        """Rough USD cost — Sonnet 4.6 pricing ($3/M in, $15/M out)."""
        return round(
            (self._total_input_tokens  / 1_000_000) * 3.0 +
            (self._total_output_tokens / 1_000_000) * 15.0,
            4,
        )

    def _map_analysis(self, raw: dict) -> FrameAnalysis:
        a = FrameAnalysis()
        a.is_frc_match        = raw.get("is_frc_match")
        a.frc_confidence      = float(raw.get("frc_confidence", 0.0))
        a.match_phase         = raw.get("match_phase")
        a.error_flag          = raw.get("error_flag")
        a.camera_angle        = raw.get("camera_angle")

        sb = raw.get("scoreboard", {})
        a.red_score           = sb.get("red_score")
        a.blue_score          = sb.get("blue_score")
        a.timer               = sb.get("timer")
        a.red_teams           = [int(t) for t in sb.get("red_teams", []) if t]
        a.blue_teams          = [int(t) for t in sb.get("blue_teams", []) if t]

        fc = raw.get("field_context", {})
        a.visible_robots      = fc.get("visible_robots")
        a.fuel_in_play        = fc.get("fuel_in_play")

        a.robot_visible       = raw.get("target_robot_visible")
        a.robot_confidence    = float(raw.get("robot_confidence", 0.0))
        a.bumper_number       = raw.get("bumper_number")
        a.bumper_number_conf  = float(raw.get("bumper_number_confidence", 0.0))
        a.bumper_alliance     = raw.get("bumper_alliance")

        gp = raw.get("robot_grid_position")
        if gp and "row" in gp and "col" in gp:
            a.robot_grid_position = (int(gp["row"]), int(gp["col"]))

        robot = raw.get("target_robot", {})
        a.has_fuel            = robot.get("has_fuel")
        a.near_hub            = robot.get("near_hub")
        a.near_opponent_zone  = robot.get("near_opponent_zone")
        a.near_outpost        = robot.get("near_outpost")
        a.is_moving           = robot.get("is_moving")
        a.is_climbing         = robot.get("is_climbing")
        a.is_defending        = robot.get("is_defending")
        a.intake_type         = robot.get("intake_type")
        a.appears_stuck       = robot.get("appears_stuck")
        a.stuck_on            = robot.get("stuck_on")
        a.just_scored         = robot.get("just_scored")
        a.scoring_location    = robot.get("scoring_location")
        a.robot_action        = robot.get("robot_action")
        return a


# ─── Google Vision Provider (stub) ───────────────────────────────────

class GoogleVisionProvider(VisionProvider):
    """
    Stub for Google Cloud Vision API.

    To implement:
      1. pip install google-cloud-vision
      2. export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
      3. Implement analyze_frame() using Vision API label/object detection
      4. Implement identify_robot() using Vision API text detection (OCR)
      5. Map results to FrameAnalysis fields
      6. Run: python frc_scout_v3.py --provider google

    Note: Google Vision has no understanding of FRC game context.
    You will need to map raw labels/objects to game concepts manually.
    Claude's approach (contextual understanding via prompt) is likely
    more accurate for role/behavior fields.
    Google Vision may be more accurate for: OCR (bumper numbers),
    object detection (fuel balls, field elements).
    """

    def __init__(self):
        raise NotImplementedError(
            "GoogleVisionProvider not yet implemented. See docstring."
        )

    def name(self) -> str:
        return "Google Cloud Vision"

    def analyze_frame(self, frame, frame_number, alliance, context) -> FrameAnalysis:
        raise NotImplementedError

    def identify_robot(self, frame, bbox, known_teams) -> tuple[int | None, float]:
        raise NotImplementedError


# ─── Combined Provider ────────────────────────────────────────────────

class CombinedVisionProvider(VisionProvider):
    """
    Runs Claude and Google Vision in parallel.
    Fields where both agree → keep at averaged confidence.
    Fields where they disagree → null (safer than picking a side).
    Only usable once GoogleVisionProvider is implemented.
    """

    def __init__(self):
        self.claude = ClaudeVisionProvider()
        self.google = GoogleVisionProvider()

    def name(self) -> str:
        return "Combined (Claude + Google Vision)"

    def analyze_frame(self, frame, frame_number, alliance, context) -> FrameAnalysis:
        ca = self.claude.analyze_frame(frame, frame_number, alliance, context)
        ga = self.google.analyze_frame(frame, frame_number, alliance, context)
        return self._merge(ca, ga)

    def identify_robot(self, frame, bbox, known_teams) -> tuple[int | None, float]:
        ct, cc = self.claude.identify_robot(frame, bbox, known_teams)
        gt, gc = self.google.identify_robot(frame, bbox, known_teams)
        if ct is not None and ct == gt:
            return ct, (cc + gc) / 2  # agreement → higher confidence
        if ct is not None and gt is None:
            return ct, cc * 0.85      # only Claude → slight penalty
        if gt is not None and ct is None:
            return gt, gc * 0.85      # only Google → slight penalty
        return None, 0.0              # disagreement → null

    def _merge(self, ca: FrameAnalysis, ga: FrameAnalysis) -> FrameAnalysis:
        m = FrameAnalysis()
        if ca.is_frc_match and ga.is_frc_match:
            m.is_frc_match   = True
            m.frc_confidence = (ca.frc_confidence + ga.frc_confidence) / 2
        else:
            m.is_frc_match   = False
            m.frc_confidence = 0.5
            m.error_flag     = "provider_disagreement" if ca.is_frc_match != ga.is_frc_match else "not_frc"
            return m.apply_gates()

        m.red_score  = _agree(ca.red_score,  ga.red_score)
        m.blue_score = _agree(ca.blue_score, ga.blue_score)
        m.timer      = ca.timer or ga.timer

        if ca.robot_visible and ga.robot_visible:
            m.robot_visible    = True
            m.robot_confidence = min(ca.robot_confidence, ga.robot_confidence)
        else:
            m.robot_visible = False
            m.error_flag    = "low_visibility"
            return m.apply_gates()

        m.has_fuel           = _agree(ca.has_fuel,           ga.has_fuel)
        m.near_hub           = _agree(ca.near_hub,           ga.near_hub)
        m.near_opponent_zone = _agree(ca.near_opponent_zone, ga.near_opponent_zone)
        m.near_outpost       = _agree(ca.near_outpost,       ga.near_outpost)
        m.is_moving          = _agree(ca.is_moving,          ga.is_moving)
        m.is_climbing        = _agree(ca.is_climbing,        ga.is_climbing)
        m.is_defending       = _agree(ca.is_defending,       ga.is_defending)
        m.intake_type        = _agree(ca.intake_type,        ga.intake_type)
        m.appears_stuck      = _agree(ca.appears_stuck,      ga.appears_stuck)
        m.stuck_on           = _agree(ca.stuck_on,           ga.stuck_on)
        m.just_scored        = _agree(ca.just_scored,        ga.just_scored)
        return m.apply_gates()


def _agree(a: Any, b: Any) -> Any:
    """Return value if both providers agree, else None."""
    if a is None and b is None:
        return None
    return a if a == b else None


# ─── Provider Registry ───────────────────────────────────────────────
PROVIDERS: dict[str, type[VisionProvider]] = {
    "claude":   ClaudeVisionProvider,
    "google":   GoogleVisionProvider,
    "combined": CombinedVisionProvider,
}


# ════════════════════════════════════════════════════════════════════
# TBA API
# ════════════════════════════════════════════════════════════════════

def tba_get(endpoint: str) -> dict | list | None:
    if not TBA_KEY:
        return None
    try:
        r = requests.get(
            f"{TBA_BASE}/{endpoint}",
            headers={"X-TBA-Auth-Key": TBA_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[!] TBA error: {e}")
        return None


def get_match_teams(event_key: str, match_number: int) -> dict:
    data = tba_get(f"match/{event_key}_qm{match_number}")
    if not data:
        return {"red": [], "blue": []}
    return {
        side: [int(t.replace("frc", "")) for t in data["alliances"][side]["team_keys"]]
        for side in ("red", "blue")
    }


def get_official_scores(event_key: str, match_number: int) -> dict | None:
    data = tba_get(f"match/{event_key}_qm{match_number}")
    if not data or not data.get("score_breakdown"):
        return None
    sb = data["score_breakdown"]
    return {
        "red_total":  sb.get("red",  {}).get("totalPoints"),
        "blue_total": sb.get("blue", {}).get("totalPoints"),
    }


# ════════════════════════════════════════════════════════════════════
# BUMPER OCR UTILITIES (Tesseract fallback only)
# ════════════════════════════════════════════════════════════════════

def _tesseract_read_bumper(upscaled_crop: np.ndarray) -> int | None:
    """Tesseract fallback — only called when Claude is uncertain."""
    gray   = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    raw    = pytesseract.image_to_string(bin, config=config).strip()
    digits = "".join(filter(str.isdigit, raw))
    if digits and 1 <= int(digits) <= 9999:
        return int(digits)
    return None


def _tba_fuzzy_match(ocr: int | None, known: list[int]) -> int | None:
    """Correct single-digit OCR misreads against known TBA team list."""
    if ocr is None or not known:
        return ocr
    if ocr in known:
        return ocr
    for team in known:
        s1, s2 = str(ocr), str(team)
        if len(s1) == len(s2) and sum(a != b for a, b in zip(s1, s2)) == 1:
            return team
    return None


class BumperVoter:
    """
    Collects identity readings across pre-match frames.
    Claude is tried first, Tesseract as fallback, TBA for validation.
    Votes across frames to find the most consistent result.
    """

    def __init__(self, known_teams: list[int], provider: VisionProvider):
        self.known    = known_teams
        self.provider = provider
        self.reads:   list[int] = []

    def add(self, frame: np.ndarray, bbox: tuple):
        team, conf = self.provider.identify_robot(frame, bbox, self.known)
        if team is not None and conf >= MIN_BUMPER_CONF:
            self.reads.append(team)

    def result(self) -> tuple[int | None, float]:
        """Returns (team_number, confidence) or (None, 0.0) if no consensus."""
        if not self.reads:
            return None, 0.0
        top, votes = Counter(self.reads).most_common(1)[0]
        conf       = votes / len(self.reads)
        return (top, conf) if conf >= MIN_OCR_VOTES else (None, conf)


# ════════════════════════════════════════════════════════════════════
# STATE ACCUMULATOR
# Only counts observations that passed confidence gates.
# Never derives fields from null data.
# ════════════════════════════════════════════════════════════════════

class RobotStateAccumulator:

    def __init__(self):
        # Behavioral counters
        self.total_obs           = 0
        self.hub_obs             = 0
        self.opponent_obs        = 0
        self.outpost_obs         = 0
        self.defending_obs       = 0
        self.moving_obs          = 0
        self.climbing_obs        = 0
        self.stuck_obs           = 0
        self.stuck_fuel_obs      = 0
        self.stuck_bump_obs      = 0
        self.fuel_obs            = 0
        self.intake_ground_obs   = 0
        self.intake_outpost_obs  = 0
        self.scored_obs          = 0
        self.scored_moving_obs   = 0
        self.high_hub_obs        = 0
        self.low_hub_obs         = 0

        # Cycle detection
        self.cycle_count         = 0
        self.had_fuel_last       = None  # None = unknown, never guess

        # Grid position history for role inference
        self.grid_positions: list[tuple] = []

        # Scoreboard readings for TBA cross-check
        self.red_score_reads:    list[int] = []
        self.blue_score_reads:   list[int] = []

        # Quality counters
        self.total_frames        = 0
        self.not_frc_frames      = 0
        self.low_vis_frames      = 0
        self.provider_errors     = 0

    def update(self, a: FrameAnalysis):
        self.total_frames += 1

        if a.error_flag == "provider_error":
            self.provider_errors += 1
            return

        if not a.is_frc_match:
            self.not_frc_frames += 1
            return

        # Collect scoreboard regardless of robot visibility
        if a.red_score  is not None: self.red_score_reads.append(a.red_score)
        if a.blue_score is not None: self.blue_score_reads.append(a.blue_score)

        if not a.robot_visible or a.robot_confidence < MIN_ROBOT_CONF:
            self.low_vis_frames += 1
            return

        # All gates passed — accumulate
        self.total_obs += 1

        if a.near_hub           is True: self.hub_obs            += 1
        if a.near_opponent_zone is True: self.opponent_obs       += 1
        if a.near_outpost       is True: self.outpost_obs        += 1
        if a.is_defending       is True: self.defending_obs      += 1
        if a.is_moving          is True: self.moving_obs         += 1
        if a.is_climbing        is True: self.climbing_obs       += 1
        if a.appears_stuck      is True: self.stuck_obs          += 1
        if a.stuck_on == "fuel":         self.stuck_fuel_obs     += 1
        if a.stuck_on == "bump":         self.stuck_bump_obs     += 1
        if a.has_fuel           is True: self.fuel_obs           += 1
        if a.intake_type == "ground":    self.intake_ground_obs  += 1
        if a.intake_type == "outpost":   self.intake_outpost_obs += 1
        if a.just_scored        is True: self.scored_obs         += 1
        if a.scoring_location == "high_hub": self.high_hub_obs  += 1
        if a.scoring_location == "low_hub":  self.low_hub_obs   += 1

        if a.robot_grid_position:
            self.grid_positions.append(a.robot_grid_position)

        # Cycle: had fuel → lost fuel near hub
        if self.had_fuel_last is not None and a.has_fuel is not None:
            lost_fuel = self.had_fuel_last and not a.has_fuel
            if lost_fuel and a.near_hub is True:
                self.cycle_count += 1
                if a.is_moving is True:
                    self.scored_moving_obs += 1

        if a.has_fuel is not None:
            self.had_fuel_last = a.has_fuel

    def derive_final_fields(self, official_scores: dict | None = None) -> dict:
        fields: dict[str, Any] = {}

        fields["_quality"] = {
            "valid_obs":       self.total_obs,
            "not_frc_frames":  self.not_frc_frames,
            "low_vis_frames":  self.low_vis_frames,
            "provider_errors": self.provider_errors,
            "total_frames":    self.total_frames,
        }

        if self.total_obs < MIN_OBS:
            fields["_warning"] = (
                f"Only {self.total_obs} valid observations (need {MIN_OBS}). "
                f"All behavioral fields null."
            )
            for f in (
                "cycling","scoring","feeding","defending","immobile",
                "intake_type_ground","intake_type_outpost","beached_on_fuel",
                "beached_on_bump","scores_while_moving","auto_climb",
                "estimated_cycles","primary_scoring_location",
            ):
                fields[f] = None
        else:
            t = self.total_obs
            hub_pct     = self.hub_obs      / t
            opp_pct     = self.opponent_obs / t
            defend_pct  = self.defending_obs / t
            stuck_pct   = self.stuck_obs    / t
            moving_pct  = self.moving_obs   / t
            climb_pct   = self.climbing_obs / t
            g_pct       = self.intake_ground_obs  / t
            o_pct       = self.intake_outpost_obs / t

            defending = defend_pct > 0.30 and opp_pct > 0.25
            cycling   = self.cycle_count >= 3 and hub_pct > 0.15
            scoring   = hub_pct > 0.50 and self.cycle_count < 3 and not defending
            feeding   = not defending and not cycling and not scoring and moving_pct > 0.35
            immobile  = stuck_pct > 0.80 and moving_pct < 0.15

            # Beaching — stuck but not fully immobile
            # Claude now reports stuck_on directly so we use those counts too
            beach_fuel = (
                (self.stuck_fuel_obs / t > 0.30 or (stuck_pct > 0.60 and hub_pct < 0.10))
                and not immobile
            )
            beach_bump = (
                (self.stuck_bump_obs / t > 0.30 or (stuck_pct > 0.60 and opp_pct < 0.10))
                and not immobile and not beach_fuel
            )

            # Primary scoring location
            if self.high_hub_obs > self.low_hub_obs:
                primary_location = "high_hub" if self.high_hub_obs > 0 else None
            elif self.low_hub_obs > 0:
                primary_location = "low_hub"
            else:
                primary_location = None

            fields.update({
                "cycling":                 cycling   or None,
                "scoring":                 scoring   or None,
                "feeding":                 feeding   or None,
                "defending":               defending or None,
                "immobile":                immobile  or None,
                "beached_on_fuel":         beach_fuel or None,
                "beached_on_bump":         beach_bump or None,
                "scores_while_moving":     (self.scored_moving_obs > 0) or None,
                "auto_climb":              (climb_pct > 0.15) or None,
                "estimated_cycles":        self.cycle_count if self.cycle_count > 0 else None,
                "intake_type_ground":      (g_pct > 0.15) or None,
                "intake_type_outpost":     (o_pct > 0.10) or None,
                "primary_scoring_location": primary_location,
            })

        # ── Scores: vision readings + TBA cross-check ─────────────
        vis_red  = Counter(self.red_score_reads).most_common(1)[0][0]  \
                   if self.red_score_reads  else None
        vis_blue = Counter(self.blue_score_reads).most_common(1)[0][0] \
                   if self.blue_score_reads else None

        fields["scores"] = {
            "vision_red":    vis_red,
            "vision_blue":   vis_blue,
            "official_red":  official_scores.get("red_total")  if official_scores else None,
            "official_blue": official_scores.get("blue_total") if official_scores else None,
            "source":        "tba" if official_scores else "vision_only",
        }

        if official_scores and vis_red is not None:
            gap_r = abs((vis_red  or 0) - (official_scores.get("red_total")  or 0))
            gap_b = abs((vis_blue or 0) - (official_scores.get("blue_total") or 0))
            if gap_r > 10 or gap_b > 10:
                fields["scores"]["_tba_mismatch"] = (
                    f"Vision vs TBA gap — RED: {gap_r} pts, BLUE: {gap_b} pts. "
                    f"Use TBA official scores."
                )

        return fields


# ════════════════════════════════════════════════════════════════════
# STREAM + ROBOT SELECTION
# ════════════════════════════════════════════════════════════════════

def resolve_stream(url: str) -> str:
    if Path(url).is_file():
        return url
    print("[*] Resolving stream with yt-dlp...")
    r = subprocess.run(
        ["yt-dlp", "--get-url", "-f", "best[height<=720]", "--no-playlist", url],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        print(f"[!] yt-dlp failed: {r.stderr.strip()}")
        sys.exit(1)
    return r.stdout.strip().split("\n")[0]


def select_robot(
    tracked,
    voters: dict[int, BumperVoter],
    frame: np.ndarray,
) -> tuple[int | None, int | None, float]:
    """
    Score every tracked robot and pick the easiest one to track.
    Robots with confirmed identity score highest.
    Returns (tracker_id, team_number, ocr_confidence).
    """
    fh, fw     = frame.shape[:2]
    all_bboxes = [d.xyxy[0] for d in tracked]
    best_tid, best_team, best_conf, best_score = None, None, 0.0, -1.0

    for det in tracked:
        tid         = det.tracker_id
        x1,y1,x2,y2 = det.xyxy[0]
        voter       = voters.get(tid)
        team, conf  = voter.result() if voter else (None, 0.0)

        area  = (x2-x1) * (y2-y1)
        cx    = ((x1+x2)/2) / fw
        cy    = ((y1+y2)/2) / fh
        dist  = ((cx-0.5)**2 + (cy-0.5)**2) ** 0.5

        # Overlap penalty
        overlap = sum(
            _iou((x1,y1,x2,y2), other)
            for other in all_bboxes
            if not all(a == b for a, b in zip((x1,y1,x2,y2), other))
        )

        score = (
            min(area / (fh * fw) * 30, 3.0)   # size reward
            + max(0, 2.0 - dist * 4)           # center reward
            + max(0, 1.5 - overlap * 5)        # occlusion penalty
            + (5.0 if team else 0.0)           # identity bonus
        )

        if score > best_score:
            best_score = score
            best_tid   = tid
            best_team  = team
            best_conf  = conf

    return best_tid, best_team, best_conf


def _iou(a: tuple, b: tuple) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0


# ════════════════════════════════════════════════════════════════════
# MAIN RUN LOOP
# ════════════════════════════════════════════════════════════════════

def run(
    url: str,
    event_key: str,
    match_number: int,
    preferred_alliance: str | None,
    provider_name: str,
):
    print("""
  ╔════════════════════════════════════════════════╗
  ║   FRC REBUILT 2026 — Grounded Scout v3        ║
  ║   Null > guess. Always.                        ║
  ╚════════════════════════════════════════════════╝
    """)

    # ── Init provider ──────────────────────────────────────────────
    if provider_name not in PROVIDERS:
        print(f"[!] Unknown provider: {provider_name}. Choose: {list(PROVIDERS)}")
        sys.exit(1)
    try:
        provider = PROVIDERS[provider_name]()
    except NotImplementedError as e:
        print(f"[!] Provider not ready: {e}")
        sys.exit(1)
    print(f"[*] Vision provider: {provider.name()}")

    # ── TBA: match roster ──────────────────────────────────────────
    print(f"[*] Fetching match {match_number} teams from TBA...")
    alliance_teams = get_match_teams(event_key, match_number)
    if alliance_teams["red"]:
        print(f"    RED:  {alliance_teams['red']}")
        print(f"    BLUE: {alliance_teams['blue']}")
    else:
        print("    [!] No TBA data — identity validation disabled")

    scout_alliance = preferred_alliance or random.choice(["red", "blue"])
    known_teams    = alliance_teams.get(scout_alliance, [])
    print(f"[*] Scouting: {scout_alliance.upper()}")

    # ── Open stream + YOLO + ByteTrack ────────────────────────────
    direct_url = resolve_stream(url)
    cap        = cv2.VideoCapture(direct_url)
    if not cap.isOpened():
        print("[!] Failed to open stream")
        sys.exit(1)

    yolo    = YOLO(YOLO_WEIGHTS)
    tracker = sv.ByteTracker()

    # ── Pre-match: identity window ─────────────────────────────────
    print(f"\n[*] Pre-match identity window ({PRE_MATCH_S}s)...")
    voters: dict[int, BumperVoter] = {}
    start = time.time()

    while time.time() - start < PRE_MATCH_S:
        ok, frame = cap.read()
        if not ok:
            break
        results = yolo(frame, verbose=False)[0]
        tracked = tracker.update_with_detections(
            sv.Detections.from_ultralytics(results)
        )
        for det in tracked:
            tid = det.tracker_id
            if tid not in voters:
                voters[tid] = BumperVoter(known_teams, provider)
            voters[tid].add(frame, det.xyxy[0])

        elapsed = time.time() - start
        if int(elapsed) % 5 == 0:
            print(f"    {elapsed:.0f}s — {len(voters)} robots seen")

    # ── Select robot to track ─────────────────────────────────────
    print("\n[*] Selecting robot...")
    ok, frame = cap.read()
    if not ok:
        print("[!] Stream ended before match")
        sys.exit(1)

    results = yolo(frame, verbose=False)[0]
    tracked = tracker.update_with_detections(
        sv.Detections.from_ultralytics(results)
    )

    best_tid, best_team, best_conf = select_robot(tracked, voters, frame)

    if best_tid is None:
        print("[!] No robots found — exiting")
        sys.exit(1)

    if best_team:
        print(f"[+] Target: team {best_team} (identity conf {best_conf:.0%})")
    else:
        print(f"[+] Target: tracker #{best_tid} (bumper unreadable)")
        if known_teams:
            best_team = random.choice(known_teams)
            print(f"    Random TBA assignment: team {best_team}")

    # ── Main tracking loop ────────────────────────────────────────
    print("\n[*] Match in progress — tracking...\n")
    state        = RobotStateAccumulator()
    frame_count  = 0
    last_vision  = 0.0
    context      = (
        f"Match {match_number} at {event_key}. "
        f"Tracking {scout_alliance.upper()} robot, team {best_team}. "
        f"Provider: {provider.name()}."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Attempt reconnect for live streams
                print("[!] Frame read failed — attempting reconnect in 5s...")
                time.sleep(5)
                try:
                    direct_url = resolve_stream(url)
                    cap.release()
                    cap = cv2.VideoCapture(direct_url)
                    continue
                except Exception:
                    print("[!] Reconnect failed — ending session")
                    break

            frame_count += 1
            now = time.time()

            # Run YOLO + ByteTrack every frame (cheap)
            results = yolo(frame, verbose=False)[0]
            tracked = tracker.update_with_detections(
                sv.Detections.from_ultralytics(results)
            )

            # Call vision provider on interval
            if now - last_vision >= VISION_INTERVAL:
                last_vision = now
                ts  = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Frame #{frame_count} → {provider.name()}...", end=" ")

                analysis = provider.analyze_frame(
                    frame, frame_count, scout_alliance, context
                )
                state.update(analysis)

                # Status line
                if analysis.is_frc_match and analysis.robot_visible:
                    action = analysis.robot_action or "unknown"
                    phase  = analysis.match_phase or "?"
                    print(f"[{phase}] {action[:70]}")
                elif not analysis.is_frc_match:
                    print(f"[NOT FRC — {analysis.error_flag}]")
                else:
                    print(f"[robot not visible — {analysis.error_flag}]")

    except KeyboardInterrupt:
        print("\n[*] Stopped by user")
    finally:
        cap.release()

    # ── Derive fields + TBA scores ────────────────────────────────
    print("\n[*] Deriving scouting fields...")
    official = get_official_scores(event_key, match_number)
    fields   = state.derive_final_fields(official)

    # ── Cost estimate (Claude only) ───────────────────────────────
    if hasattr(provider, "cost_estimate"):
        fields["_cost_usd"] = provider.cost_estimate()

    # ── Save output ───────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"match_{match_number}_team_{best_team}_{ts}.json"

    session = {
        "event_key":    event_key,
        "match_number": match_number,
        "alliance":     scout_alliance,
        "team_number":  best_team,
        "provider":     provider.name(),
        "identity_conf": round(best_conf, 2),
        "scouting_fields": fields,
    }
    with open(out_path, "w") as f:
        json.dump(session, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  Match {match_number} | Team {best_team} | {scout_alliance.upper()}")
    print(f"{'─'*52}")
    for k, v in fields.items():
        if not k.startswith("_"):
            print(f"  {k:<32} {v}")
    if "_cost_usd" in fields:
        print(f"\n  Estimated API cost: ${fields['_cost_usd']:.4f}")
    print(f"  Saved → {out_path}\n")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRC REBUILT 2026 — Grounded Scout v3"
    )
    parser.add_argument("--url",      required=True,
                        help="YouTube/Twitch URL or local MP4 path")
    parser.add_argument("--event",    required=True,
                        help="TBA event key, e.g. 2026onsc")
    parser.add_argument("--match",    required=True, type=int,
                        help="Qualification match number")
    parser.add_argument("--alliance", choices=["red", "blue"], default=None,
                        help="Alliance to scout (random if omitted)")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="claude",
                        help="Vision provider (default: claude)")
    args = parser.parse_args()

    run(args.url, args.event, args.match, args.alliance, args.provider)

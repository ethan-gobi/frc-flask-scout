# FRC Flask Livestream Scout

A local-first Python Flask project for automated FRC livestream scouting using OpenCV, Ultralytics YOLO, SQLite, and pandas.

## Features

- Read livestream/video/local file input with OpenCV (`/save_match_setup`, `/start_tracking`, `/stop_tracking`)
- Accept standard YouTube watch URLs (`youtube.com/watch?...`) and short URLs (`youtu.be/...`) by resolving to direct media streams via `yt-dlp` before OpenCV
- Detect objects each frame with YOLO (Ultralytics)
- Estimate required scouting fields with `{ value, confidence }`
- Store event snapshots in local SQLite (`data/scouting.db`)
- Export stored scouting records to CSV/XLSX in `data/exports/`
- Flask dashboard at `/`

## Project structure

- `app.py`
- `detector.py`
- `tracker.py`
- `events.py`
- `exports.py`
- `storage.py`
- `templates/index.html`
- `requirements.txt`
- `README.md`

## Supported scouting fields

- cycling
- scoring
- feeding
- defending
- immobile
- intake_type_ground
- intake_type_outpost_or_source
- traversal_trench
- traversal_bump
- auto_climb_status
- beached_on_fuel
- beached_on_bump
- scores_while_moving
- robot_broke
- estimated_points_scored

Each field is emitted with:

- `value`
- `confidence`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open: http://127.0.0.1:5000


## Run examples by source type

### Live stream
- Save setup with a live source URL (for example RTSP/HLS), then click **Start Tracking**.

### Old YouTube livestream / recorded YouTube video
- Paste the normal YouTube URL into **Stream URL or Video Path**.
- The app resolves it with `yt-dlp` before opening in OpenCV.
- Tracking begins only when you click **Start Tracking**.

### Local MP4 file
- Enter a file path like `/home/user/matches/qm12.mp4` (or equivalent path on your machine).
- Save setup, then click **Start Tracking**.


## Step-by-step: run and interact with every part

### 1) Start from the project folder

```bash
cd /workspace/frc-flask-scout
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the Flask app

```bash
python app.py
```

You should see Flask start on `http://127.0.0.1:5000`.

### 5) Open the dashboard (`/`)

Go to:

- `http://127.0.0.1:5000/`

From the dashboard you can interact with each major part:

1. **System Status**
   - View setup status (`No match configured` / `Match configured`).
   - View tracking status (`Tracking stopped` / `Tracking running`).
   - View source status (`Recorded source loaded` / `Live source loaded`).
2. **Match Setup and Tracking**
   - Enter `Match ID` and `Stream URL or Video Path`.
   - Optional: pick `source_mode` (`auto`, `live`, `recorded`, `local_file`).
   - Click **Save Match Setup** (this does not start tracking).
   - Click **Start Tracking** to begin processing.
   - Click **Stop Tracking** to stop processing and keep the match configured.
   - Click **End Match** to finalize and clear active match setup.
3. **Latest Estimated Scouting Fields**
   - View all required fields with live `value` and `confidence`.
4. **Export**
   - Click **Download CSV** or **Download XLSX**.
   - Files are generated/saved in `data/exports/`.

### 6) Interact with each API endpoint directly (optional)

Use another terminal while the server is running.

#### A. Dashboard/status

```bash
curl http://127.0.0.1:5000/
curl http://127.0.0.1:5000/status
```

#### B. Match + tracking lifecycle

Save setup:

```bash
curl -X POST http://127.0.0.1:5000/save_match_setup \
  -H "Content-Type: application/json" \
  -d '{"match_number":1,"source_input":"https://www.youtube.com/watch?v=o-d2D77V3f4","source_mode":"auto"}'
```

Start tracking:

```bash
curl -X POST http://127.0.0.1:5000/start_tracking
```

Stop tracking:

```bash
curl -X POST http://127.0.0.1:5000/stop_tracking
```

End match:

```bash
curl -X POST http://127.0.0.1:5000/end_match
```

#### D. Export endpoints

```bash
curl -OJ http://127.0.0.1:5000/export/csv
curl -OJ http://127.0.0.1:5000/export/xlsx
```

### 7) Where data is stored

- SQLite database: `data/scouting.db`
- Export files: `data/exports/`

### 8) Typical session order (recommended)

1. Start app.
2. Save setup (`/save_match_setup`) with match ID + source input.
3. Start tracking (`/start_tracking`).
4. Let it run during the match or recorded file.
5. Stop tracking (`/stop_tracking`) if needed.
6. End match (`/end_match`).
7. Export CSV/XLSX (`/export/csv`, `/export/xlsx`).



#### YouTube URL support details

When `/start_tracking` opens a configured YouTube watch URL (`youtube.com/watch?...`) or short URL (`youtu.be/...`), the app first resolves it to a direct playable media URL by running:

```bash
python -m yt_dlp -g <youtube_url>
```

The first non-empty output line is then passed to OpenCV `VideoCapture`. Non-YouTube URLs are sent to OpenCV unchanged.

#### Stream troubleshooting

If stream start fails:

1. Confirm `yt-dlp` is installed in the same environment as Flask:
   ```bash
   python -m pip show yt-dlp
   ```
2. Test URL resolution manually:
   ```bash
   python -m yt_dlp -g "https://www.youtube.com/watch?v=o-d2D77V3f4"
   ```
3. If URL resolution succeeds but OpenCV still cannot open it, your local OpenCV/FFmpeg build may not support the returned stream format; try another quality/source URL or update your OpenCV runtime.

## Gated tracking workflow (required)

Tracking does **not** begin automatically.

You must do this in order:

1. Save setup with both:
   - Match ID (`match_number`)
   - Stream URL or video path (`source_input`)
2. Click **Start Tracking** (`/start_tracking`).

If either field is missing, the app returns clean errors:
- `Match ID is required`
- `Stream URL or video path is required`

### Source types supported

- Live stream URLs (e.g., RTSP/RTMP/HLS)
- YouTube live URLs
- YouTube past livestream recordings / normal video URLs
- Local video files (e.g., `/path/to/video.mp4`)

### Source mode

Optional `source_mode` values:
- `auto` (default)
- `live`
- `recorded`
- `local_file`

### Recorded-video behavior

For `recorded` and `local_file` sources, processing runs frame-by-frame without enforced real-time sleep, so it can run faster than live pacing depending on hardware and decode speed.

### API examples for the gated workflow

Save match setup (required before tracking):

```bash
curl -X POST http://127.0.0.1:5000/save_match_setup \
  -H "Content-Type: application/json" \
  -d '{"match_number": 1, "source_input": "https://www.youtube.com/watch?v=o-d2D77V3f4", "source_mode": "auto"}'
```

Start tracking explicitly:

```bash
curl -X POST http://127.0.0.1:5000/start_tracking
```

Stop tracking (keeps match configured):

```bash
curl -X POST http://127.0.0.1:5000/stop_tracking
```

End match (finalizes and clears active setup):

```bash
curl -X POST http://127.0.0.1:5000/end_match
```

## API endpoints

- `GET /` dashboard
- `POST /save_match_setup` body/json: `match_number`, `source_input`, optional `source_mode`
- `POST /start_tracking`
- `POST /stop_tracking`
- `POST /end_match`
- `POST /start_match` (alias to `/save_match_setup`)
- `POST /start_stream` (returns guidance to use `/start_tracking`)
- `POST /stop_stream` (alias to `/stop_tracking`)
- `GET /export/csv`
- `GET /export/xlsx`
- `GET /status`

## Example curl calls

```bash
curl -X POST http://127.0.0.1:5000/save_match_setup -H "Content-Type: application/json" -d '{"match_number":1,"source_input":"/tmp/match.mp4","source_mode":"local_file"}'
curl -X POST http://127.0.0.1:5000/start_tracking
curl -X POST http://127.0.0.1:5000/stop_tracking
curl -X POST http://127.0.0.1:5000/end_match
curl -O http://127.0.0.1:5000/export/csv
```

## Notes

- The default YOLO weights (`yolov8n.pt`) may auto-download on first run.
- Heuristic field estimation is intentionally simple and local; tune `_estimate_fields` in `tracker.py` for your team workflow.
- All data remains local on disk.

## Current limitations and expected accuracy

This project is a **baseline prototype**, not a production-grade autonomous scout yet.

### Key limitations

- **No robot identity persistence**: detections are frame-level only; the app does not maintain stable robot IDs across long sequences or occlusions.
- **Generic YOLO weights**: default `yolov8n.pt` is COCO-trained and not FRC-specific, so game-piece/field-context understanding is limited.
- **Heuristic field inference**: many scouting outputs are inferred from simple detection counts/confidence, not from explicit event recognition (e.g., real scoring events, climb verification, traversal geometry).
- **Confidence is not calibrated probability**: confidence values are useful relative signals, but they are not guaranteed to match true likelihood.
- **Camera-dependent performance**: stream angle, lighting, compression, bitrate, motion blur, and partial occlusion can significantly impact reliability.
- **No per-field model validation pipeline**: there is no built-in labeled-dataset evaluation script yet (precision/recall/F1 by field).
- **Schema scope**: only the requested fields are currently modeled; items commonly present in manual scouting forms (e.g., driver ability ratings, freeform notes semantics) are not auto-estimated.

### Practical accuracy expectations (before FRC-specific training)

Without domain-specific training data, realistic expectations are:

- **Robot presence / rough activity signals**: low-to-moderate reliability in clean broadcast views.
- **Role-style labels** (`cycling`, `feeding`, `defending`, etc.): moderate at best, highly scenario-dependent.
- **Fine-grained tactical states** (`beached_on_*`, `scores_while_moving`, ~`robot_broke`): often noisy without tailored vision logic and temporal rules.
- **Point estimation** (`estimated_points_scored`): coarse heuristic only; should not be treated as official or high-confidence scoring.

As a rough baseline, expect **high variance** and frequently a broad band like **~40–70% agreement** with human scouts depending on stream quality and match dynamics. You should treat this as an assistant signal and keep human review in the loop.

### How to improve accuracy

1. Build an FRC-labeled dataset (per field/event), including difficult edge cases.
2. Fine-tune detection/classification models for your season/game objects.
3. Add temporal tracking + event state machines (instead of single-frame heuristics).
4. Calibrate confidence scores with validation data.
5. Report per-field metrics (precision/recall/F1) on held-out matches.


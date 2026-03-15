# FRC Flask Livestream Scout

A local-first Python Flask project for automated FRC livestream scouting using OpenCV, Ultralytics YOLO, SQLite, and pandas.

## Features

- Read livestream/video URL with OpenCV (`/start_stream`, `/stop_stream`)
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
   - View whether stream processing is running.
   - View current active match metadata.
2. **Start/End Match**
   - Enter `match_number` and `team_number`.
   - Click **Start Match** to begin storing events under that match.
   - Click **End Match** to stop and finalize that match record.
3. **Stream Controls**
   - Enter a stream/video URL (examples: `rtsp://...`, direct `http://...`/`https://...`, local file path supported by OpenCV, or normal YouTube watch links).
   - Click **Start Stream** to begin frame capture + inference loop.
   - Click **Stop Stream** to stop capture and processing.
4. **Latest Estimated Scouting Fields**
   - View all required fields with live `value` and `confidence`.
5. **Export**
   - Click **Download CSV** or **Download XLSX**.
   - Files are generated/saved in `data/exports/`.

### 6) Interact with each API endpoint directly (optional)

Use another terminal while the server is running.

#### A. Dashboard/status

```bash
curl http://127.0.0.1:5000/
curl http://127.0.0.1:5000/status
```

#### B. Match lifecycle

Start match:

```bash
curl -X POST -F "match_number=1" -F "team_number=254" http://127.0.0.1:5000/start_match
```

End match:

```bash
curl -X POST http://127.0.0.1:5000/end_match
```

#### C. Stream lifecycle

Start stream:

```bash
curl -X POST -F "stream_url=https://example.com/stream.m3u8" http://127.0.0.1:5000/start_stream
```

Stop stream:

```bash
curl -X POST http://127.0.0.1:5000/stop_stream
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
2. Start a match (`/start_match`).
3. Start stream (`/start_stream`).
4. Let it run during the match.
5. Stop stream (`/stop_stream`).
6. End match (`/end_match`).
7. Export CSV/XLSX (`/export/csv`, `/export/xlsx`).



#### YouTube URL support details

When `/start_stream` receives a YouTube watch URL (`youtube.com/watch?...`) or short URL (`youtu.be/...`), the app first resolves it to a direct playable media URL by running:

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

## API endpoints

- `GET /` dashboard
- `POST /start_match` body/form: `match_number`, `team_number`
- `POST /end_match`
- `POST /start_stream` body/form: `stream_url`
- `POST /stop_stream`
- `GET /export/csv`
- `GET /export/xlsx`
- `GET /status`

## Example curl calls

```bash
curl -X POST -F "match_number=1" -F "team_number=254" http://127.0.0.1:5000/start_match
curl -X POST -F "stream_url=https://example.com/stream.m3u8" http://127.0.0.1:5000/start_stream
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
- **Fine-grained tactical states** (`beached_on_*`, `scores_while_moving`, `robot_broke`): often noisy without tailored vision logic and temporal rules.
- **Point estimation** (`estimated_points_scored`): coarse heuristic only; should not be treated as official or high-confidence scoring.

As a rough baseline, expect **high variance** and frequently a broad band like **~40–70% agreement** with human scouts depending on stream quality and match dynamics. You should treat this as an assistant signal and keep human review in the loop.

### How to improve accuracy

1. Build an FRC-labeled dataset (per field/event), including difficult edge cases.
2. Fine-tune detection/classification models for your season/game objects.
3. Add temporal tracking + event state machines (instead of single-frame heuristics).
4. Calibrate confidence scores with validation data.
5. Report per-field metrics (precision/recall/F1) on held-out matches.


import logging

from flask import Flask, jsonify, render_template, request, send_file

from events import SCOUTING_FIELDS, empty_scouting_payload
from exports import export_csv, export_xlsx
from tracker import StreamTracker

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
tracker = StreamTracker()


@app.route("/")
def dashboard():
    latest = tracker.get_latest_snapshot()
    state = tracker.get_state()
    return render_template(
        "index.html",
        fields=SCOUTING_FIELDS,
        active_match=tracker.active_match,
        stream_running=tracker.stream_running,
        latest=latest,
        state=state,
    )


@app.route("/save_match_setup", methods=["POST"])
def save_match_setup():
    payload = request.get_json(silent=True) or request.form or {}
    raw_match_number = payload.get("match_number", payload.get("match_id", 0))
    match_number = int(raw_match_number or 0)
    source_input = (payload.get("source_input") or payload.get("stream_url") or "").strip()
    source_mode = (payload.get("source_mode") or "auto").strip() or "auto"
    match_type = (payload.get("match_type") or "qualification").strip() or "qualification"

    if match_number <= 0:
        return jsonify({"ok": False, "error": "Match number is required"}), 400
    if not source_input:
        return jsonify({"ok": False, "error": "Stream URL or video path is required"}), 400

    try:
        match_id = tracker.configure_match(
            match_number=match_number,
            source_input=source_input,
            source_mode=source_mode,
            match_type=match_type,
        )
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({
        "ok": True,
        "message": "Match configured",
        "match_number": match_number,
        "internal_record_id": match_id,
        "match_id": match_id,
    })


@app.route("/start_tracking", methods=["POST"])
def start_tracking():
    try:
        resolved_url = tracker.start_tracking()
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    state = tracker.get_state()
    return jsonify(
        {
            "ok": True,
            "message": "Tracking running",
            "resolved_url": resolved_url,
            "source_status": state["source_status"],
        }
    )


@app.route("/stop_tracking", methods=["POST"])
def stop_tracking():
    tracker.stop_stream()
    return jsonify({"ok": True, "message": "Tracking stopped"})


@app.route("/end_match", methods=["POST"])
def end_match():
    if not tracker.active_match:
        return jsonify({"ok": False, "error": "No active match"}), 400

    summary = tracker.end_match()
    return jsonify({"ok": True, "summary": summary, "message": "Match ended"})


# Backward-compatible aliases for previous API names.
@app.route("/start_match", methods=["POST"])
def start_match_alias():
    return save_match_setup()


@app.route("/start_stream", methods=["POST"])
def start_stream_alias():
    return jsonify({"ok": False, "error": "Use /start_tracking after /save_match_setup"}), 400


@app.route("/stop_stream", methods=["POST"])
def stop_stream_alias():
    return stop_tracking()


@app.route("/status")
def status():
    return jsonify(
        {
            "ok": True,
            "stream_running": tracker.stream_running,
            "active_match": tracker.active_match,
            "latest": tracker.get_latest_snapshot() or empty_scouting_payload(),
            "state": tracker.get_state(),
        }
    )


@app.route("/export/csv")
def export_data_csv():
    path = export_csv()
    return send_file(path, as_attachment=True)


@app.route("/export/xlsx")
def export_data_xlsx():
    path = export_xlsx()
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)

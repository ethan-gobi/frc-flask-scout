from flask import Flask, jsonify, render_template, request, send_file

from events import SCOUTING_FIELDS, empty_scouting_payload
from exports import export_csv, export_xlsx
from tracker import StreamTracker

app = Flask(__name__)
tracker = StreamTracker()


@app.route("/")
def dashboard():
    latest = tracker.get_latest_snapshot()
    return render_template(
        "index.html",
        fields=SCOUTING_FIELDS,
        active_match=tracker.active_match,
        stream_running=tracker.stream_running,
        latest=latest,
    )


@app.route("/start_match", methods=["POST"])
def start_match():
    payload = request.get_json(silent=True) or request.form or {}
    match_number = int(payload.get("match_number", 0))
    team_number = int(payload.get("team_number", 0))

    if match_number <= 0 or team_number <= 0:
        return jsonify({"ok": False, "error": "match_number and team_number must be positive integers"}), 400

    match_id = tracker.start_match(match_number=match_number, team_number=team_number)
    return jsonify({"ok": True, "match_id": match_id})


@app.route("/end_match", methods=["POST"])
def end_match():
    if not tracker.active_match:
        return jsonify({"ok": False, "error": "No active match"}), 400

    summary = tracker.end_match()
    return jsonify({"ok": True, "summary": summary})


@app.route("/start_stream", methods=["POST"])
def start_stream():
    payload = request.get_json(silent=True) or request.form or {}
    stream_url = payload.get("stream_url")
    if not stream_url:
        return jsonify({"ok": False, "error": "stream_url is required"}), 400

    try:
        tracker.start_stream(stream_url)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "message": "Stream started"})


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    tracker.stop_stream()
    return jsonify({"ok": True, "message": "Stream stopped"})


@app.route("/status")
def status():
    return jsonify(
        {
            "ok": True,
            "stream_running": tracker.stream_running,
            "active_match": tracker.active_match,
            "latest": tracker.get_latest_snapshot() or empty_scouting_payload(),
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

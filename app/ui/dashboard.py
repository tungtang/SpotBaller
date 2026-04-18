from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SpotBaller · Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles (Streamlit-compatible) ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1200px; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    .stCaption { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

SHORT_LABELS = {
    "a basketball player shooting the ball toward the hoop": "Shooting",
    "a basketball player passing the ball to a teammate": "Passing",
    "a basketball player dribbling the basketball": "Dribbling",
    "a basketball player defending another player": "Defending",
}


def fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_optional(url: str) -> dict | list | None:
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def api_ok(base: str) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=5)
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("API base URL", value="http://127.0.0.1:8000").rstrip("/")
    if api_ok(api_base):
        st.success("API reachable")
    else:
        st.warning("Cannot reach `/health` — start the API (`uvicorn app.api.main:app`).")
    st.markdown(
        f"[Web app home]({api_base}/) · [All results]({api_base}/results) (API jobs + CLI runs) — video, stats, JSON."
    )

    st.divider()
    st.header("Processing")
    mode = st.selectbox("Mode", options=["local", "cloud"], index=0, help="Local runs in-process; cloud uses the RQ worker.")
    weights = st.text_input("Detector weights", value="auto", help="'auto' uses E-BARD when `models/e-bard/BODD_yolov8n_0001.pt` exists.")
    use_pretrained = st.checkbox("Pretrained stack (SigLIP + TrOCR + action hints)", value=True)
    use_videomae = st.checkbox("VideoMAE auxiliary (heavy)", value=False)
    auto_poll = st.checkbox("Auto-refresh while running", value=True)
    poll_seconds = st.slider("Refresh interval (s)", min_value=1, max_value=10, value=2)

    with st.expander("Recommended models (from API)", expanded=False):
        try:
            recs = fetch_json(f"{api_base}/models/recommendations")
            for r in recs:
                st.markdown(f"**{r.get('task', '?')}**  \n`{r.get('model_name', '')}`")
                st.caption(r.get("notes", ""))
        except Exception as exc:
            st.caption(f"Could not load: {exc}")

st.title("SpotBaller")
st.caption(
    "Upload game footage, run the full detection → tracking → identity → stats pipeline, "
    "and review per-player and team outputs."
)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    video_file = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "avi"], help="Sideline or full-court footage works best.")
with col_b:
    st.write("")
    st.write("")
    start = st.button("Analyze video", type="primary", disabled=video_file is None, use_container_width=True)
with col_c:
    st.write("")
    st.write("")
    if st.session_state.job_id:
        if st.button("Clear job", use_container_width=True):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.session_state.video_path = None
            st.rerun()


def start_job() -> None:
    files = {"file": (video_file.name, video_file.getvalue(), video_file.type or "video/mp4")}
    upload_resp = requests.post(f"{api_base}/videos", files=files, timeout=300)
    upload_resp.raise_for_status()
    upload_payload = upload_resp.json()
    st.session_state.video_path = upload_payload["path"]
    job_resp = requests.post(
        f"{api_base}/jobs",
        data={
            "video_path": st.session_state.video_path,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": "true" if use_pretrained else "false",
            "use_videomae": "true" if use_videomae else "false",
        },
        timeout=600,
    )
    job_resp.raise_for_status()
    job_payload = job_resp.json()
    st.session_state.job_id = job_payload["job_id"]
    st.session_state.job_status = job_payload.get("status", "queued")


if start and video_file is not None:
    try:
        start_job()
        st.success(f"Started job `{st.session_state.job_id}`")
    except Exception as exc:
        st.error(f"Failed to start: {exc}")


if not st.session_state.job_id:
    st.info("Upload a video and click **Analyze video** to begin.")
    st.stop()


st.divider()
st.subheader("Job status")

status_ph = st.empty()
bar_ph = st.empty()
detail_ph = st.empty()

try:
    job_payload = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}")
except Exception as exc:
    st.error(f"Failed to load job: {exc}")
    st.stop()

st.session_state.job_status = job_payload.get("db_status", job_payload.get("status"))
status = str(st.session_state.job_status or "unknown").lower()

progress_map = {"queued": 12, "retrying": 45, "processing": 70, "done": 100, "failed": 100}
status_ph.markdown(f"**Status:** `{status}` · **Job ID:** `{st.session_state.job_id}`")
bar_ph.progress(progress_map.get(status, 5))

if status in {"queued", "processing", "retrying"}:
    detail_ph.caption("Processing: detection (YOLO/E-BARD) → ByteTrack → identity → events → stats.")
else:
    with detail_ph.expander("Raw job payload", expanded=False):
        st.json(job_payload)

c1, c2 = st.columns(2)
with c1:
    if st.button("Refresh status"):
        st.rerun()
with c2:
    if auto_poll and status in {"queued", "processing", "retrying"}:
        time.sleep(float(poll_seconds))
        st.rerun()

if status != "done":
    if status == "failed":
        st.error("Job failed. Check API logs and `runtime/jobs/<id>/job.json`.")
    st.stop()

report_url = f"{api_base}/results/{st.session_state.job_id}"
st.info(f"[Open this analysis in your browser]({report_url}) — watch annotated video and inspect exports.")

# --- Results ---
result = job_payload.get("result") or {}
pipeline = result.get("pipeline") or {}
frames_n = result.get("frames_processed")
weights_used = result.get("weights", job_payload.get("weights"))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Frames processed", frames_n if frames_n is not None else "—")
with m2:
    st.metric("Detector", Path(str(weights_used)).name if weights_used else "—")
with m3:
    tr = pipeline.get("tracker_backend", "—")
    st.metric("Tracker", tr)
with m4:
    ps = pipeline.get("pretrained_stack") or {}
    loaded = sum(bool(ps.get(k)) for k in ("siglip_loaded", "trocr_loaded", "videomae_loaded"))
    st.metric("HF heads active", f"{loaded}/3")

tab_stats, tab_team, tab_ai, tab_dl = st.tabs(
    ["Player stats", "Team box score", "AI pipeline & hints", "Downloads"],
)

stats_rows = result.get("stats")
if not stats_rows:
    try:
        stats_rows = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}/stats")
    except Exception:
        stats_rows = []

team_stats = result.get("team_box_score")
if team_stats is None:
    try:
        team_stats = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}/team-stats")
    except Exception:
        team_stats = {}

with tab_stats:
    if stats_rows:
        df = pd.DataFrame(stats_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "player_id": st.column_config.NumberColumn("Player ID", format="%d"),
                "minutes_on_court": st.column_config.NumberColumn("Min", format="%.2f"),
                "fg_pct": st.column_config.NumberColumn("FG%", format="%.1f"),
                "pts": st.column_config.NumberColumn("PTS", format="%d"),
            },
        )
    else:
        st.warning("No stats rows returned.")

with tab_team:
    if isinstance(team_stats, dict) and team_stats:
        st.json(team_stats)
    else:
        st.caption("No team box score available.")

with tab_ai:
    st.markdown("##### Pipeline metadata")
    st.json(pipeline if pipeline else {"message": "No pipeline block in result (older job?)."})

    err = (pipeline.get("pretrained_stack") or {}).get("load_errors") or []
    if err:
        st.warning("Some pretrained components failed to load:")
        for e in err:
            st.text(e)

    hints = result.get("action_hints") or []
    st.markdown("##### SigLIP action hints (sampled frames)")
    if hints:
        rows = []
        for h in hints:
            fi = h.get("frame_index", 0)
            scores = h.get("siglip_action_scores") or {}
            for prompt, p in scores.items():
                label = SHORT_LABELS.get(prompt, prompt[:40] + "…")
                rows.append({"frame": fi, "category": label, "probability": p})
        if rows:
            hdf = pd.DataFrame(rows)
            wide = hdf.pivot_table(index="frame", columns="category", values="probability", aggfunc="first")
            st.line_chart(wide, height=280)
            with st.expander("Per-frame scores (table)"):
                st.dataframe(hdf.sort_values("frame"), use_container_width=True, hide_index=True)
    else:
        st.caption("No action hints (disable pretrained stack or check SigLIP load errors).")

    vma = result.get("videomae_aux") or []
    st.markdown("##### VideoMAE auxiliary (Kinetics logits)")
    if vma:
        st.caption("Top Kinetics classes from 16-frame windows — auxiliary only, not basketball-specific.")
        st.dataframe(pd.DataFrame(vma), use_container_width=True, hide_index=True)
    else:
        st.caption("VideoMAE disabled or no windows emitted.")

with tab_dl:
    st.markdown(
        f"**Web report:** [open in browser]({report_url}) (same data as below, with embedded video)."
    )
    st.markdown("Download JSON exports from the job directory on the API host, or use the bundles below.")
    j_stats = json.dumps(stats_rows, indent=2) if stats_rows else "{}"
    j_team = json.dumps(team_stats, indent=2) if team_stats else "{}"
    j_full = json.dumps(result, indent=2) if result else json.dumps(job_payload, indent=2)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "stats.json",
            data=j_stats,
            file_name=f"{st.session_state.job_id}_stats.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "team_box_score.json",
            data=j_team,
            file_name=f"{st.session_state.job_id}_team_box_score.json",
            mime="application/json",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "full_result.json",
            data=j_full,
            file_name=f"{st.session_state.job_id}_full_result.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(
        f"Artifact paths (on server): `runtime/jobs/{st.session_state.job_id}/` — "
        "includes `annotated.mp4`, `tracks.json`, `events.json`, `pipeline.json`, `action_hints.json`."
    )

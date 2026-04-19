from __future__ import annotations

import json
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


def render_local_job_controls(api_base: str, job_id: str, jp: dict) -> None:
    """Stop / Start / Resume / Delete for local API jobs (uses current job payload)."""
    if str(jp.get("mode", "local")).lower() != "local":
        return
    status = str(jp.get("db_status", jp.get("status")) or "unknown").lower()
    worker_active = bool(jp.get("worker_active"))
    can_stop = status == "processing" and worker_active
    can_start = status in ("stopped", "failed") or (status == "processing" and not worker_active)

    st.subheader("Job controls")
    jc1, jc2, jc3 = st.columns(3)
    with jc1:
        if st.button(
            "Stop",
            disabled=not can_stop,
            help="Ask the worker to stop after the current frame.",
            use_container_width=True,
            key=f"btn_job_stop_{job_id}",
        ):
            try:
                sr = requests.post(f"{api_base}/jobs/{job_id}/stop", timeout=60)
                sr.raise_for_status()
                st.success("Stop requested — status will update shortly.")
            except Exception as exc:
                st.error(str(exc))
            st.rerun()
    with jc2:
        if st.button(
            "Start / Resume",
            disabled=not can_start,
            help="Re-run the full pipeline from the beginning of the same video (reuses this job id).",
            use_container_width=True,
            key=f"btn_job_start_{job_id}",
        ):
            try:
                sr = requests.post(f"{api_base}/jobs/{job_id}/start", timeout=120)
                sr.raise_for_status()
                st.success("Analysis started.")
            except Exception as exc:
                st.error(str(exc))
            st.rerun()
    with jc3:
        if st.button(
            "Delete job",
            type="secondary",
            use_container_width=True,
            key=f"btn_job_delete_{job_id}",
        ):
            try:
                dr = requests.delete(f"{api_base}/jobs/{job_id}", timeout=120)
                if dr.status_code == 404:
                    st.warning("Job was already removed.")
                else:
                    dr.raise_for_status()
                st.session_state.job_id = None
                st.session_state.job_status = None
                st.session_state.video_path = None
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.caption(
        "Cloud jobs use the worker queue — these buttons apply to **local** jobs only. "
        "Resume runs a full pass again from frame 0."
    )


def _lifecycle_title(job_payload: dict, status: str) -> str:
    """Human-readable outcome: Complete vs Stopped vs Failed vs Running (from API `lifecycle`)."""
    lc = (job_payload.get("lifecycle") or "").lower()
    if lc == "complete":
        return "Complete"
    if lc == "stopped":
        return "Stopped"
    if lc == "failed":
        return "Failed"
    if lc == "running":
        return "Running"
    if status == "done":
        return "Complete"
    if status == "stopped":
        return "Stopped"
    if status == "failed":
        return "Failed"
    return status.replace("_", " ").title()


def render_job_status_ui(
    job_payload: dict,
    progress_data: dict | list | None,
    status: str,
    job_id: str,
) -> None:
    """Status line, progress bar, frame/completion metrics, caption or raw JSON expander."""
    pct_bar: float | None = None
    if isinstance(progress_data, dict):
        p = progress_data.get("percent_complete")
        if isinstance(p, (int, float)):
            pct_bar = min(1.0, max(0.0, float(p) / 100.0))
        elif status in {"queued", "processing", "retrying"}:
            pct_bar = 0.1
    elif status == "queued":
        pct_bar = 0.05
    elif status in {"processing", "retrying"}:
        pct_bar = 0.08

    fallback_map = {
        "queued": 0.12,
        "retrying": 0.45,
        "processing": 0.7,
        "done": 1.0,
        "failed": 1.0,
        "stopped": 1.0,
    }
    title = _lifecycle_title(job_payload, status)
    st.markdown(f"**{title}** · raw `{status}` · **Job ID:** `{job_id}`")
    mode_j = str(job_payload.get("mode", "")).lower()
    res = job_payload.get("result") if isinstance(job_payload.get("result"), dict) else {}
    vm_name = job_payload.get("gcp_vm") or res.get("remote_vm")
    vm_zone = job_payload.get("gcp_zone") or res.get("remote_zone")
    if mode_j == "vm" and vm_name:
        zone_bit = f" · `{vm_zone}`" if vm_zone else ""
        st.caption(f"GCP VM: **{vm_name}**{zone_bit}")
    st.progress(pct_bar if pct_bar is not None else fallback_map.get(status, 0.05))

    prog_cols = st.columns(2)
    with prog_cols[0]:
        if isinstance(progress_data, dict):
            cur_f = progress_data.get("frames_processed")
            tot_f = progress_data.get("total_frames")
            if isinstance(cur_f, int) and isinstance(tot_f, int) and tot_f > 0:
                st.metric("Frames", f"{cur_f:,} / {tot_f:,}")
            elif isinstance(cur_f, int):
                st.metric("Frames", f"{cur_f:,} (total unknown)")
            else:
                st.metric("Frames", "—")
        else:
            st.metric("Frames", "—")
    with prog_cols[1]:
        if isinstance(progress_data, dict):
            p_pct = progress_data.get("percent_complete")
            if isinstance(p_pct, (int, float)):
                st.metric("Completion", f"{float(p_pct):.1f}%")
            else:
                st.metric("Completion", "—")
        else:
            st.metric("Completion", "—")

    if status in {"queued", "processing", "retrying"}:
        st.caption("Processing: detection (YOLO/E-BARD) → ByteTrack → identity → events → stats.")
    elif status == "stopped":
        st.warning(
            job_payload.get("stop_detail")
            or "Analysis stopped before completion (e.g. API restarted). Re-run the video to finish."
        )
        with st.expander("Raw job payload", expanded=False):
            st.json(job_payload)
    else:
        with st.expander("Raw job payload", expanded=False):
            st.json(job_payload)


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
    mode = st.selectbox(
        "Run analysis on",
        options=["local", "vm", "cloud", "gcp"],
        index=0,
        format_func=lambda m: {
            "local": "This machine (API process)",
            "vm": "GCP VM (gcloud scp + ssh)",
            "cloud": "Redis / RQ worker",
            "gcp": "GCS + Pub/Sub GPU pool",
        }.get(m, m),
        help="Local = same host as the API. VM = run on the instance named in SPOTBALLER_GCLOUD_VM (e.g. spotballer-vm-2); requires gcloud + SPOTBALLER_GCLOUD_* on the API host.",
    )
    gcp_cfg = fetch_optional(f"{api_base}/config/gcp") if api_ok(api_base) else None
    if mode == "vm" and isinstance(gcp_cfg, dict) and gcp_cfg.get("vm"):
        st.caption(
            f"Target VM from API env: **{gcp_cfg['vm']}**"
            + (f" (`{gcp_cfg['zone']}`)" if gcp_cfg.get("zone") else "")
        )
    elif mode == "vm":
        st.caption("Set SPOTBALLER_GCLOUD_VM / ZONE / PROJECT on the API host, or `/config/gcp` will be empty.")
    max_frames_ui = st.number_input(
        "Max frames (0 = full video)",
        min_value=0,
        value=0,
        help="Limit for quick tests; same cap is passed to the VM when using GCP VM mode.",
    )
    weights = st.text_input("Detector weights", value="auto", help="'auto' uses E-BARD when `models/e-bard/BODD_yolov8n_0001.pt` exists.")
    use_pretrained = st.checkbox("Pretrained stack (SigLIP + TrOCR + action hints)", value=True)
    use_videomae = st.checkbox("VideoMAE auxiliary (heavy)", value=False)
    if mode == "vm":
        with st.expander("GCP VM requirements", expanded=False):
            st.markdown(
                "On the **API machine**, set `SPOTBALLER_GCLOUD_VM`, `SPOTBALLER_GCLOUD_ZONE`, "
                "`SPOTBALLER_GCLOUD_PROJECT`, and install `gcloud`. Use `CLOUDSDK_PYTHON` if the SDK needs Python 3.10+."
            )
    auto_poll = st.checkbox(
        "Auto-refresh while running",
        value=True,
        help="Updates only the job status block (not the whole page) to avoid flicker.",
    )
    poll_seconds = st.slider("Refresh interval (s)", min_value=2, max_value=15, value=4)

    with st.expander("Recommended models (from API)", expanded=False):
        try:
            recs = fetch_json(f"{api_base}/models/recommendations")
            for r in recs:
                st.markdown(f"**{r.get('task', '?')}**  \n`{r.get('model_name', '')}`")
                st.caption(r.get("notes", ""))
        except Exception as exc:
            st.caption(f"Could not load: {exc}")

    st.divider()
    with st.expander("Compare two jobs (performance.json)", expanded=False):
        cj_a = st.text_input("Job ID A", key="cmp_job_a", placeholder="local run")
        cj_b = st.text_input("Job ID B", key="cmp_job_b", placeholder="VM run")
        if st.button("Compare performance", key="btn_cmp_perf"):
            if not (cj_a and cj_b):
                st.warning("Enter both job IDs.")
            else:
                try:
                    r = requests.get(
                        f"{api_base}/jobs/compare-performance",
                        params={"job_a": cj_a.strip(), "job_b": cj_b.strip()},
                        timeout=30,
                    )
                    r.raise_for_status()
                    cmp_data = r.json()
                    pa, pb = cmp_data.get("performance_a"), cmp_data.get("performance_b")
                    if pa is None:
                        st.error(f"No performance.json for job A (`{cj_a}`).")
                    elif pb is None:
                        st.error(f"No performance.json for job B (`{cj_b}`).")
                    else:
                        rows = cmp_data.get("rows") or []
                        if rows:
                            cdf = pd.DataFrame(rows)
                            st.dataframe(cdf, use_container_width=True, hide_index=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.json(pa)
                        with c2:
                            st.json(pb)
                except Exception as exc:
                    st.error(str(exc))

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
    mf_str = "" if int(max_frames_ui) <= 0 else str(int(max_frames_ui))
    job_resp = requests.post(
        f"{api_base}/jobs",
        data={
            "video_path": st.session_state.video_path,
            "mode": mode,
            "weights": weights,
            "use_pretrained_stack": "true" if use_pretrained else "false",
            "use_videomae": "true" if use_videomae else "false",
            "max_frames": mf_str,
        },
        timeout=7200 if mode == "vm" else 120,
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

try:
    job_payload = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}")
except Exception as exc:
    st.error(f"Failed to load job: {exc}")
    st.stop()

st.session_state.job_status = job_payload.get("db_status", job_payload.get("status"))
status = str(st.session_state.job_status or "unknown").lower()
running = status in {"queued", "processing", "retrying"}

if not (running and auto_poll):
    render_local_job_controls(api_base, st.session_state.job_id, job_payload)

if running and auto_poll:
    # Fragment reruns on a timer only for this block — avoids full-page flicker from st.rerun().
    @st.fragment(run_every=float(poll_seconds))
    def _job_status_live() -> None:
        try:
            jp = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}")
            pr = fetch_optional(f"{api_base}/jobs/{st.session_state.job_id}/progress")
        except Exception as exc:
            st.error(f"Failed to load job: {exc}")
            return
        s = str(jp.get("db_status", jp.get("status")) or "unknown").lower()
        render_local_job_controls(api_base, st.session_state.job_id, jp)
        render_job_status_ui(jp, pr, s, st.session_state.job_id)
        if s in ("done", "failed", "stopped"):
            st.rerun()

    _job_status_live()
    st.stop()

progress_data = fetch_optional(f"{api_base}/jobs/{st.session_state.job_id}/progress")
render_job_status_ui(job_payload, progress_data, status, st.session_state.job_id)

if running and not auto_poll:
    if st.button("Refresh status"):
        st.rerun()
    st.stop()

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
    ["Player stats (by jersey)", "Team box score", "AI pipeline & hints", "Downloads"],
)

stats_rows = result.get("stats")
if not stats_rows:
    try:
        stats_rows = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}/stats")
    except Exception:
        stats_rows = []

stats_by_track = result.get("stats_by_track")
if not stats_by_track:
    stats_by_track = fetch_optional(f"{api_base}/jobs/{st.session_state.job_id}/stats/by-track")

team_stats = result.get("team_box_score")
if team_stats is None:
    try:
        team_stats = fetch_json(f"{api_base}/jobs/{st.session_state.job_id}/team-stats")
    except Exception:
        team_stats = {}

with tab_stats:
    st.caption(
        "Rows are **merged by detected jersey number** (and team). Multiple tracker IDs mapping to the same "
        "jersey are summed. Unresolved OCR rows stay separate per track."
    )
    if stats_rows:
        df = pd.DataFrame(stats_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "identity_kind": st.column_config.TextColumn("Identity"),
                "jersey_key": st.column_config.TextColumn("Jersey key"),
                "jersey_number": st.column_config.TextColumn("Jersey #"),
                "player_label": st.column_config.TextColumn("Player"),
                "team_name": st.column_config.TextColumn("Team"),
                "player_number_confidence": st.column_config.NumberColumn("Jersey conf", format="%.2f"),
                "merged_track_count": st.column_config.NumberColumn("Tracks merged", format="%d"),
                "minutes_on_court": st.column_config.NumberColumn("Min", format="%.2f"),
                "fg_pct": st.column_config.NumberColumn("FG%", format="%.1f"),
                "pts": st.column_config.NumberColumn("PTS", format="%d"),
            },
        )
    else:
        st.warning("No stats rows returned.")

    with st.expander("Raw per-track stats (debug only)", expanded=False):
        st.caption(
            "Internal MOT `player_id` rows with explicit `jersey_number` / `jersey_key` columns for "
            "identity debugging."
        )
        if stats_by_track:
            tdf = pd.DataFrame(stats_by_track)
            track_cols = [
                c
                for c in (
                    "player_id",
                    "jersey_number",
                    "jersey_key",
                    "player_number",
                    "player_number_confidence",
                    "player_label",
                    "team_name",
                    "minutes_on_court",
                    "touches",
                    "pts",
                    "fg_pct",
                )
                if c in tdf.columns
            ]
            if track_cols:
                st.dataframe(tdf[track_cols], use_container_width=True, hide_index=True)
            else:
                st.dataframe(tdf, use_container_width=True, hide_index=True)
        else:
            st.caption("No `stats_by_track` for this job (re-run analysis to generate).")

with tab_team:
    if isinstance(team_stats, list) and team_stats:
        for team in team_stats:
            team_name = team.get("team_name", "Unknown Team")
            st.markdown(f"##### {team_name}")
            totals = team.get("team_totals", {}) or {}
            tm1, tm2, tm3, tm4, tm5 = st.columns(5)
            tm1.metric("PTS", int(totals.get("pts", 0)))
            tm2.metric("REB", int(totals.get("reb", 0)))
            tm3.metric("AST", int(totals.get("ast", 0)))
            tm4.metric("TO", int(totals.get("tov", 0)))
            tm5.metric("Touches", int(totals.get("touches", 0)))

            players = team.get("players", []) or []
            if players:
                pdf = pd.DataFrame(players)
                cols = [c for c in ("player_number", "player_label", "minutes_on_court", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg_pct") if c in pdf.columns]
                st.dataframe(
                    pdf[cols] if cols else pdf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "player_number": st.column_config.TextColumn("Jersey #"),
                        "minutes_on_court": st.column_config.NumberColumn("Min", format="%.2f"),
                        "fg_pct": st.column_config.NumberColumn("FG%", format="%.1f"),
                    },
                )
            st.divider()
    else:
        st.caption("No team box score available.")

with tab_ai:
    st.markdown("##### Pipeline summary")
    if pipeline:
        pp = pipeline.get("pretrained_stack", {}) or {}
        perf = pipeline.get("performance", {}) or {}
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Weights", Path(str(pipeline.get("detection_weights", "—"))).name)
        pm2.metric("Tracker", str(pipeline.get("tracker_backend", "—")))
        pm3.metric("Pretrained loaded", f"{sum(bool(pp.get(k)) for k in ('siglip_loaded', 'trocr_loaded', 'videomae_loaded'))}/3")
        pm4.metric("Effective FPS", perf.get("fps_effective", "—"))

        stage = perf.get("stage_s", {}) or {}
        if stage:
            st.markdown("###### Stage time breakdown (seconds)")
            sdf = pd.DataFrame([{"stage": k, "seconds": float(v)} for k, v in stage.items()]).sort_values("seconds", ascending=False)
            st.bar_chart(sdf.set_index("stage"))
            st.dataframe(sdf, use_container_width=True, hide_index=True)
    else:
        st.caption("No pipeline block in result (older job?).")

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
            top_actions = (
                hdf.groupby("category", as_index=False)["probability"]
                .mean()
                .sort_values("probability", ascending=False)
                .rename(columns={"probability": "avg_probability"})
            )
            st.markdown("###### Average action confidence")
            st.dataframe(top_actions, use_container_width=True, hide_index=True)
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
    j_track = json.dumps(stats_by_track, indent=2) if stats_by_track else "{}"
    j_team = json.dumps(team_stats, indent=2) if team_stats else "{}"
    j_full = json.dumps(result, indent=2) if result else json.dumps(job_payload, indent=2)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "stats.json (by jersey)",
            data=j_stats,
            file_name=f"{st.session_state.job_id}_stats.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "stats_by_track.json",
            data=j_track,
            file_name=f"{st.session_state.job_id}_stats_by_track.json",
            mime="application/json",
            use_container_width=True,
            disabled=not stats_by_track,
        )
    with d3:
        st.download_button(
            "team_box_score.json",
            data=j_team,
            file_name=f"{st.session_state.job_id}_team_box_score.json",
            mime="application/json",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            "full_result.json",
            data=j_full,
            file_name=f"{st.session_state.job_id}_full_result.json",
            mime="application/json",
            use_container_width=True,
        )
    c1, c2 = st.columns(2)
    with c1:
        if isinstance(team_stats, list) and team_stats:
            team_rows = []
            for t in team_stats:
                for p in t.get("players", []) or []:
                    team_rows.append({"team_name": t.get("team_name"), **p})
            if team_rows:
                st.download_button(
                    "team_box_score_players.csv",
                    data=pd.DataFrame(team_rows).to_csv(index=False),
                    file_name=f"{st.session_state.job_id}_team_box_score_players.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with c2:
        perf = pipeline.get("performance", {}) if isinstance(pipeline, dict) else {}
        if perf:
            perf_rows = [{"metric": "elapsed_s", "value": perf.get("elapsed_s")}, {"metric": "fps_effective", "value": perf.get("fps_effective")}]
            for k, v in (perf.get("stage_s", {}) or {}).items():
                perf_rows.append({"metric": f"stage_{k}", "value": v})
            st.download_button(
                "pipeline_performance.csv",
                data=pd.DataFrame(perf_rows).to_csv(index=False),
                file_name=f"{st.session_state.job_id}_pipeline_performance.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(
        f"Artifact paths (on server): `runtime/jobs/{st.session_state.job_id}/` — "
        "`stats.json` is **by jersey**; `stats_by_track.json` is raw MOT ids for debugging."
    )

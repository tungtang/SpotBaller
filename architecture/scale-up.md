# Production scale-up plan · ~100 concurrent users

This document proposes a **production architecture** and **infrastructure** for SpotBaller-class workloads when you need to support on the order of **100 concurrent users**. It is written to be implementable on common clouds (AWS/GCP/Azure) or Kubernetes; swap managed service names as needed.

---

## 1. Clarify what “100 concurrent users” means

Capacity depends on the **dominant pattern**. Define SLOs explicitly:

| Interpretation | Typical load | Hardest resource |
|----------------|--------------|------------------|
| **A. 100 active browser sessions** | Polling status, listing jobs, occasional upload | API + DB read QPS |
| **B. 100 simultaneous analyses** | 100 full videos in `run_video_analysis` at once | GPU + queue + egress |
| **C. Mixed** | e.g. 20 heavy analyses + 80 light UI users | Blend of GPU and API |

**Default assumption for sizing below:** you want to survive **B in bursts** (short spikes toward 100 parallel jobs) while **A** is always comfortable. If your product caps concurrent analyses lower (e.g. 10), adjust worker counts accordingly.

**Suggested product guardrails:**

- Per-user **max concurrent jobs** (e.g. 1–2).
- Global **max in-flight analyses** (queue depth + worker pool) with clear “queued” UX.
- **Max upload size** and **max video duration** (or frame budget) to bound worst-case GPU time.

---

## 2. Target architecture (logical)

```
Clients (browser / mobile)
        │
        ▼
┌───────────────────┐     ┌──────────────────┐
│  CDN (optional)   │     │  WAF / DDoS      │
└─────────┬─────────┘     └────────┬─────────┘
          │                        │
          ▼                        ▼
┌─────────────────────────────────────────────┐
│  API tier (stateless)                        │
│  · AuthN/Z (JWT / session / API keys)        │
│  · Upload: presigned URLs → object storage   │
│  · Jobs: create, status, cancel, artifacts   │
│  · Rate limits + idempotency keys            │
└─────────┬───────────────────────────────────┘
          │
          ├──► OLTP DB (Postgres): users, jobs, billing, pointers
          │
          ├──► Redis (or cloud queue): job queue, locks, short-lived state
          │
          └──► Object storage (S3-compatible): raw video, outputs, logs

          ┌──────────────────────────────────────┐
          │  Worker tier (autoscaled)           │
          │  · GPU workers: YOLO + tracker +     │
          │    optional HF stack (or callout)    │
          │  · CPU workers: mux, thumbs, export  │
          └──────────────────────────────────────┘
```

**Principles:**

- **API never runs long video loops** in the request thread at scale.
- **One job = one video** (or explicit sharding strategy for very long assets).
- **Artifacts on object storage**; DB stores metadata and URLs only.
- **Idempotent job creation** to survive client retries.

---

## 3. Component responsibilities

### 3.1 API service (FastAPI or equivalent)

- **Ingress:** HTTPS, JWT/session, CORS, request size limits for non-upload paths.
- **Upload flow:** return **presigned PUT** URL to object storage; client uploads directly; API receives **complete** callback or client posts `finalize` with object key.
- **Job lifecycle:** `queued` → `processing` → `complete` | `failed` | `stopped`.
- **Status:** read `progress.json` from object storage or DB-updated progress rows (prefer DB or Redis for high poll volume).
- **Downloads:** presigned GET for `annotated.mp4`, JSON bundles, CSV exports.

**Scale:** 2–6 API replicas minimum for HA; scale on CPU and p95 latency.

### 3.2 Job queue

- **Redis + RQ** (already familiar in this repo) is viable to ~100 workers if Redis is HA and network is stable.
- At stricter production SLOs, prefer **SQS / PubSub / RabbitMQ** + dedicated worker processes with **visibility timeout** and **dead-letter queues**.

**Rule:** queue depth and **global concurrency cap** protect GPUs from overload.

### 3.3 Workers (GPU)

Each analysis job is **CPU+GPU heavy** and **long-running** (minutes).

Recommended split:

- **GPU worker image:** CUDA + PyTorch + Ultralytics + (optional) Transformers; mounts secrets for HF token if needed.
- **Concurrency per GPU:** often **1 job per GPU** for predictable latency; **2** only if profiling shows headroom and you accept tail latency.

**Rough capacity math (order of magnitude):**

- If average analysis = **5 min** on one GPU, one GPU completes **12 jobs/hour**.
- **100 parallel jobs** need on the order of **100 GPU-worker slots** for a 5-minute finish *in parallel* (burst). In practice you **queue** and give users ETA unless you provision ~100 GPUs (expensive).

**Realistic production target:** provision enough GPUs to meet **p95 queue wait + processing** SLO (e.g. “90% of jobs start within 2 minutes, finish within 15 minutes for a 1080p 40-min game cap”).

### 3.4 Workers (CPU)

Optional **CPU pool** for:

- Thumbnail / preview generation  
- Packaging ZIP exports  
- Lightweight validation (container probe, duration, resolution)

Keeps GPU workers from doing I/O-bound fluff.

### 3.5 Data stores

| Store | Data |
|-------|------|
| **Postgres** | Users, orgs, job rows, status transitions, billing meter events, artifact keys |
| **Object storage** | `uploads/*`, `jobs/{id}/annotated.mp4`, `stats.json`, `tracks.json`, etc. |
| **Redis** | Queue, rate limits, distributed locks, optional progress cache |

**Retention:** lifecycle policies on raw uploads (e.g. 30–90 days) vs outputs (longer if product requires).

### 3.6 Observability

- **Metrics:** API RPS, p95 latency, queue depth, worker utilization, GPU util/mem, job duration histograms, failure reasons.
- **Logs:** structured JSON per job id; correlation id from API → worker.
- **Tracing:** OpenTelemetry from API to worker (where supported).
- **Alerts:** queue age, error rate, stuck `processing`, disk full on workers (if local scratch).

### 3.7 Security and compliance

- Encrypt objects **at rest** (SSE-S3 or KMS).
- **TLS** everywhere; **least-privilege** IAM for workers (read/write only job prefix).
- If using **hosted HF inference**, treat game frames as **PII-sensitive**; define data residency and retention.

---

## 4. Infrastructure checklist (concrete “what to provision”)

Minimal production footprint for **HA + ~100 concurrent UI users + bounded parallel analysis**:

| Item | Suggested starting point | Notes |
|------|---------------------------|--------|
| API | 2+ instances, autoscale 2–10 | Behind load balancer |
| Postgres | Managed, Multi-AZ, backups | PITR enabled |
| Redis | Managed HA **or** cloud queue | RQ vs SQS tradeoff |
| Object storage | Versioned bucket + lifecycle | Presigned URLs |
| GPU workers | Pool autoscale 0–N on queue depth | Start N = peak parallel jobs you’ll fund |
| CPU workers | Small pool 2–8 | Exports / housekeeping |
| CDN | Optional for static + finished video | Signed URLs |
| Secrets | KMS + secret manager | HF token, DB creds |
| CI/CD | Build images per tier | **SpotBaller:** one `infra/gcp/Dockerfile.gpu-worker` image; set `SPOTBALLER_CONTAINER_ROLE=api` or `worker` per Deployment |

---

## 5. Application changes to align with this architecture

These are the main **codebase-level** steps to make the repo “production-shaped”:

1. **Presigned upload path** (remove large multipart through API except as fallback).
2. **Worker entrypoint** that pulls object key + params, runs `run_video_analysis`, writes artifacts to storage, updates DB.
3. **Progress:** write to DB or Redis in addition to (or instead of) filesystem `progress.json` when running multi-node.
4. **Cancellation:** propagate `stop_event` / cooperative cancel across worker (already present for local jobs; wire to queue message).
5. **Artifact allowlist** and path discipline (already partially present in `web_report`); enforce per-job prefixes in storage.

---

## 6. Cost and scaling levers

- **Concurrency caps** matter more than peak users: video is **O(duration × resolution × model stack)**.
- **Tiered quality:** fast path (detector + tracker only) vs full pretrained stack as a **paid** or **async** tier.
- **Regional GPUs:** largest driver of monthly cost; use **spot/preemptible** for batch-only workers if SLO allows.
- **Model offload:** optional HF Endpoints for OCR/embeddings only if network + privacy tradeoffs are acceptable.

---

## 7. Summary

Serving **100 concurrent users** is straightforward for **API and UI** with a small HA API fleet and Postgres. Serving **100 concurrent full analyses** requires a **queue**, a **GPU worker fleet sized to your SLO** (often much smaller than 100 GPUs with queuing), and **object storage** for all heavy artifacts. The architecture above separates **stateless API**, **durable queue**, **scalable workers**, and **blob storage** so you can grow from tens to hundreds of parallel jobs without redesigning the control plane.

---

## 8. Choosing **GCP · GKE GPU node pool + job queue** (Option 1)

Use this path when you want **Google-managed Kubernetes**, **first-class GPU node pools**, and a **durable queue** between API and workers—without locking into a single-vendor batch SKU (you can still run the **API** on Cloud Run and only use GKE for GPU workers).

### 8.1 Decision checklist (steps to confirm this is the right option)

1. **Job shape:** Each analysis is **minutes long**, **GPU-bound**, and **parallelizable** (one video ≈ one job). If yes, GKE + queue fits.
2. **Team skills:** You can operate **Kubernetes** (manifests, upgrades, pod debugging) or have access to someone who can. If not, evaluate **Cloud Batch** or **Vertex AI custom training/custom jobs** before GKE.
3. **GPU quota:** In GCP Console → **IAM & Admin → Quotas**, search for **GPUs** in the target region (e.g. `NVIDIA L4`, `T4`, `A100`). **Request quota increases** early; cluster creation succeeds but **scheduling fails** without GPU quota.
4. **Regional colocation:** Pick **one region** (e.g. `us-central1`) for **GKE**, **Cloud Storage**, and **Pub/Sub** to minimize latency and egress.
5. **Spot tolerance:** If variable delay and **retries on preemption** are acceptable, plan a **Spot / preemptible GPU node pool** (large savings). If not, use **on-demand** for a baseline pool and Spot for overflow only.
6. **Artifacts:** You standardize on **GCS** for uploads and outputs (`annotated.mp4`, JSON, CSV).
7. **Autoscaling signal:** You can scale workers from **queue backlog** or **oldest-unacked message age** (e.g. **KEDA** + Pub/Sub, or a custom metric exported to Cloud Monitoring).

If most items are “yes,” use the setup in §9.

### 8.2 Target topology on GCP

| Layer | GCP service | Role |
|--------|-------------|------|
| Object storage | **Cloud Storage** | Raw uploads, job outputs (`gs://…/jobs/{id}/…`) |
| Queue | **Pub/Sub** or **Cloud Tasks** | API enqueues work; workers consume |
| GPU compute | **GKE** (GPU **node pool**) | Runs `video-worker` containers |
| Images | **Artifact Registry** | Versioned `api` / `video-worker` images |
| DB | **Cloud SQL (Postgres)** | Job metadata, users (optional **Memorystore** for cache) |
| Secrets | **Secret Manager** | DB URL, `HF_TOKEN`, signing keys |
| Ingress | **HTTPS load balancer** (+ managed cert) | Public API |

**Worker pattern:** **long-lived Deployment** that pulls messages (efficient) *or* **one Kubernetes Job per message** (strong isolation, good with Spot). Pick based on how you handle **Pub/Sub ack deadlines** (§9).

---

## 9. Detailed setup steps (GCP · GKE GPU pool + queue)

Order is a practical sequence; exact flags depend on **GKE Standard vs Autopilot** and your GPU SKU. **Standard GKE** is the usual choice for explicit GPU node pools.

### Phase A — Project and APIs

1. Select a **GCP project**; link **billing**.
2. Enable APIs, e.g.  
   `container.googleapis.com`, `artifactregistry.googleapis.com`, `pubsub.googleapis.com`, `storage.googleapis.com`, `secretmanager.googleapis.com`, `sqladmin.googleapis.com` (if using Cloud SQL).

### Phase B — Registry, buckets, queue

3. Create **Artifact Registry** repository (Docker), region = your compute region.
4. Create **GCS buckets** (e.g. uploads + outputs), uniform bucket-level access, lifecycle rules as needed.
5. Create **Pub/Sub** topic `video-jobs` and subscription `video-jobs-workers`.  
   - **Important:** default Pub/Sub **ack deadline** is short relative to multi-minute GPU jobs. For production you typically either:  
     - use **short control messages** + **job state in Cloud SQL** (worker heartbeats), or  
     - use **Cloud Tasks** with a **lease** pattern, or  
     - run workers that **modify ack deadline** / use a **streaming pull** pattern with care.  
   Do not assume a single message can stay unacked for a 40-minute transcode without extra design.

### Phase C — GKE cluster and GPU node pool

6. Create a **VPC-native** GKE cluster (**Workload Identity** enabled).
7. Add a **GPU node pool**:
   - Choose **machine type** + **GPU** (`nvidia-tesla-t4`, `nvidia-l4`, etc.) per quota and benchmarks.
   - Apply **taints** (e.g. `nvidia.com/gpu=present:NoSchedule`) so only GPU workloads schedule there.
   - Enable **cluster autoscaler** with min/max nodes aligned to budget.
   - Optional: second pool using **Spot** VMs for batch.
8. Verify nodes report allocatable **`nvidia.com/gpu`** and that your GKE version’s **GPU driver / device plugin** path is satisfied (follow current Google docs for your minor version).

### Phase D — IAM

9. Create service accounts, e.g. `spotballer-worker@PROJECT.iam.gserviceaccount.com` with:
   - **Storage** read/write on job prefixes (prefer per-bucket IAM, not project editor).
   - **Secret Manager** accessor.
   - Pub/Sub **subscriber** (if the runtime identity pulls from Pub/Sub).
10. Map **Workload Identity**: Kubernetes SA ↔ GCP SA.

### Phase E — Deploy workers and API

11. Build and push **video-worker** image (CUDA + app): entrypoint wraps `run_video_analysis` with **GCS** paths (download to local SSD / streaming as you implement).
12. Deploy workers with:
    - `resources.limits: nvidia.com/gpu: "1"` (or fractional only if you deliberately share),
    - **nodeSelector** / **affinity** for GPU pool,
    - **tolerations** for GPU taints.
13. Deploy API (GKE or **Cloud Run**) to **publish** job messages and expose REST.

### Phase F — Autoscaling

14. Install **KEDA** (optional but common): scale worker **Deployment** on Pub/Sub **backlog** metric.
15. Ensure **cluster autoscaler** max nodes × GPUs ≥ peak parallel jobs you are willing to pay for.

### Phase G — Hardening

16. **Secrets:** Secret Manager + CSI driver or synced env.
17. **Observability:** Cloud Logging + **DCGM**-based GPU metrics where applicable.
18. **DLQ:** dead-letter subscription / topic for poison messages.
19. **Runbooks:** CUDA/driver mismatch, OOM, preemption, IAM denial on GCS.

**Repo implementation:** see `infra/gcp/README.md` (Dockerfile, example Deployment, env vars) and `app/gcp/` (Pub/Sub worker + GCS job runner). Create jobs with **`mode=gcp`** on the API after setting `SPOTBALLER_GCS_BUCKET` and `SPOTBALLER_PUBSUB_TOPIC`.

---

## 10. Pricing estimate · **per processed frame**

Official SKUs change by **region**, **GPU type**, and **commitment**. Use the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) before budgeting. Below is a **reusable method** plus **illustrative** math.

### 10.1 GPU cost per frame (one full GPU per job)

Let:

- \(F\) = frames processed per job  
- \(\mathrm{FPS}_\mathrm{eff}\) = sustained end-to-end **effective FPS** (from `performance.json`, e.g. `fps_effective`)  
- \(R\) = **effective** \$/hour for the **GPU worker** (VM + attached GPU + disk—use calculator)

GPU hours per job:

\[
\mathrm{GPUh} = \frac{F}{\mathrm{FPS}_\mathrm{eff} \times 3600}
\]

GPU cost per job: \(R \times \mathrm{GPUh}\).  
**GPU cost per frame:**

\[
\mathrm{\$/frame} = \frac{R}{\mathrm{FPS}_\mathrm{eff} \times 3600}
\]

Longer videos cost more **total** dollars because \(F\) increases linearly; **per-frame** GPU cost is flat if \(\mathrm{FPS}_\mathrm{eff}\) is stable.

### 10.2 Illustrative table (verify R in calculator)

Assume **R = $1.00 / GPU-hour** (placeholder—replace with your SKU):

| FPS_eff | Approx. GPU $ / **1M frames** |
|--------|---------------------------------|
| 10 | ~ $27.8 |
| 2 | ~ $139 |
| 0.5 | ~ $556 |

### 10.3 Worked example (order of magnitude)

If **R ≈ $1.20/hr** and **FPS_eff ≈ 0.437** (heavy pretrained-style run):

\[
\mathrm{\$/frame} \approx 1.20 / (0.437 \times 3600) \approx \$0.00076
\]

For **F = 174** frames → GPU-only ≈ **$0.13** at that illustrative rate (excludes GCS egress, DB, API, logging).

### 10.4 Non-GPU extras

- **GCS:** storage + **egress** (dominant if users download large MP4s out of GCP).  
- **GKE:** control plane fee (where applicable) + **non-GPU** node overhead.  
- **Cloud SQL / Memorystore:** mostly uptime-driven.  
- **Pub/Sub:** usually small next to GPU.

### 10.5 Cost levers

- **Spot / preemptible** GPU pools + **idempotent retries**.  
- **Committed use discounts** on steady baselines.  
- **Product tiers** (lighter model stack → higher FPS_eff → lower $/frame).  
- **Same-region** data and clients to avoid egress.

---

*Last updated: added GCP GKE GPU node pool + queue decision path, setup phases, Pub/Sub ack caveat, and per-frame GPU pricing methodology (verify all rates in the Google Cloud Pricing Calculator).*

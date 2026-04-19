from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline.pretrained_stack import pipeline_config_from_flags
from app.pipeline.video_analyzer import run_video_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local basketball video analytics.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", default="runtime/local_run", type=Path)
    parser.add_argument("--weights", default="auto", help="Model path or 'auto' to prefer local E-BARD.")
    parser.add_argument(
        "--no-pretrained-stack",
        action="store_true",
        help="Disable SigLIP/TrOCR/action-hint models (detector + tracker only).",
    )
    parser.add_argument(
        "--videomae",
        action="store_true",
        help="Enable VideoMAE auxiliary clip logits (heavy; requires transformers + VRAM/CPU).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N frames (for smoke tests on long videos).",
    )
    parser.add_argument(
        "--prefetch-frames",
        type=int,
        default=0,
        metavar="N",
        help="Enable frame prefetch queue with up to N buffered frames (0 disables).",
    )
    parser.add_argument(
        "--async-writer",
        action="store_true",
        help="Write annotated video on a background thread.",
    )
    parser.add_argument(
        "--identity-workers",
        type=int,
        default=0,
        metavar="N",
        help="Use N threads for identity extraction on non-pretrained path (0 disables).",
    )
    parser.add_argument(
        "--identity-stride",
        type=int,
        default=None,
        metavar="N",
        help="Identity (SigLIP/TrOCR) every N frames; omit to use SPOTBALLER_IDENTITY_STRIDE or 1.",
    )
    parser.add_argument(
        "--no-rim-fallback",
        action="store_true",
        help="Disable synthetic hoop box when the detector has no rim/hoop class (scoring needs a rim track).",
    )
    args = parser.parse_args()
    pipeline_cfg = pipeline_config_from_flags(
        use_pretrained_stack=str(not args.no_pretrained_stack).lower(),
        use_videomae=str(args.videomae).lower(),
    )
    result = run_video_analysis(
        args.video,
        args.out,
        args.weights,
        pipeline_config=pipeline_cfg,
        max_frames=args.max_frames,
        prefetch_frames=args.prefetch_frames,
        async_writer=args.async_writer,
        identity_workers=args.identity_workers,
        rim_fallback=not args.no_rim_fallback,
        identity_stride=args.identity_stride,
    )
    print(f"Processed {result['frames_processed']} frames")
    print(f"Stats saved to {args.out / 'stats.json'} and {args.out / 'stats.csv'}")
    perf = result.get("performance") or {}
    if perf:
        print(f"Elapsed: {perf.get('elapsed_s')}s | Effective FPS: {perf.get('fps_effective')}")
        print(f"Performance details: {args.out / 'performance.json'}")


if __name__ == "__main__":
    main()

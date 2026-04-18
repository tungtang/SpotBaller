from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelCandidate:
    task: str
    model_name: str
    source: str
    notes: str


def get_model_shortlist() -> list[ModelCandidate]:
    """
    Pragmatic shortlist for basketball analytics metrics.
    """
    return [
        ModelCandidate(
            task="player_ball_rim_detection",
            model_name="GabrieleGiudici/E-BARD-detection-models (BODD_yolov8n_0001.pt)",
            source="Hugging Face",
            notes="Best basketball-specific detector found; integrate first for box-score pipeline.",
        ),
        ModelCandidate(
            task="multi_object_tracking",
            model_name="ByteTrack",
            source="Open source SOTA baseline",
            notes="Strong stability for player IDs from detector outputs.",
        ),
        ModelCandidate(
            task="jersey_number_recognition",
            model_name="microsoft/trocr-base-printed (STR baseline in pipeline)",
            source="Hugging Face",
            notes="PARSeq-style role: TrOCR on player crops + tracklet voting; Tesseract fallback.",
        ),
        ModelCandidate(
            task="team_recognition",
            model_name="google/siglip-base-patch16-224 embeddings + k-means",
            source="Hugging Face",
            notes="Jersey-region SigLIP embeddings clustered to two teams; RGB fallback if sparse.",
        ),
        ModelCandidate(
            task="event_action_recognition",
            model_name="SigLIP zero-shot prompts + optional MCG-NJU/videomae-base",
            source="Hugging Face",
            notes="SigLIP text-image scores each ~12 frames; VideoMAE Kinetics logits optional auxiliary.",
        ),
    ]

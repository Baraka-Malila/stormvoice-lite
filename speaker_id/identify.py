from __future__ import annotations
import numpy as np
from sqlalchemy.orm import Session
from db.models import Speaker, SpeakerEmbedding
from features.mfcc import extract_mfcc

THRESHOLD = 0.70


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def enroll_speaker(name: str, clip_paths: list[str], db: Session) -> Speaker:
    """Average MFCC vectors from all clips and store as speaker centroid."""
    vecs = [extract_mfcc(p) for p in clip_paths]
    centroid = np.mean(vecs, axis=0)

    speaker = db.query(Speaker).filter_by(name=name).first()
    if speaker is None:
        speaker = Speaker(name=name)
        db.add(speaker)
        db.flush()

    emb = db.query(SpeakerEmbedding).filter_by(speaker_id=speaker.id).first()
    if emb is None:
        emb = SpeakerEmbedding(speaker_id=speaker.id)
        db.add(emb)
    emb.set_vector(centroid.tolist())
    emb.sample_count = len(clip_paths)
    db.commit()
    db.refresh(speaker)
    return speaker


def identify_speaker(clip_path: str, db: Session) -> tuple[str, float]:
    """Cosine similarity vs stored centroids. Returns (name, confidence)."""
    vec = extract_mfcc(clip_path)
    embeddings = db.query(SpeakerEmbedding).all()
    if not embeddings:
        return "Unknown", 0.0
    best_name, best_sim = "Unknown", 0.0
    for emb in embeddings:
        centroid = np.array(emb.get_vector(), dtype=np.float32)
        sim = _cosine(vec, centroid)
        if sim > best_sim:
            best_sim = sim
            best_name = emb.speaker.name
    if best_sim < THRESHOLD:
        return "Unknown", float(best_sim)
    return best_name, float(best_sim)

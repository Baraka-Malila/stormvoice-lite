from __future__ import annotations
import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Speaker(Base):
    __tablename__ = "speakers"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    embedding = relationship("SpeakerEmbedding", back_populates="speaker", uselist=False,
                             cascade="all, delete-orphan")
    sessions = relationship("VoiceSession", back_populates="speaker")


class SpeakerEmbedding(Base):
    __tablename__ = "speaker_embeddings"
    id = Column(Integer, primary_key=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=False)
    embedding_json = Column(Text, nullable=False)
    sample_count = Column(Integer, default=0)
    speaker = relationship("Speaker", back_populates="embedding")

    def get_vector(self) -> list[float]:
        return json.loads(self.embedding_json)

    def set_vector(self, vec: list[float]) -> None:
        self.embedding_json = json.dumps([float(v) for v in vec])


class VoiceSession(Base):
    __tablename__ = "voice_sessions"
    id = Column(Integer, primary_key=True)
    speaker_id = Column(Integer, ForeignKey("speakers.id"), nullable=True)
    transcript = Column(Text, default="")
    fraud_category = Column(String, default="Normal Request")
    risk_score = Column(Integer, default=0)
    risk_level = Column(String, default="Low")
    recommended_action = Column(String, default="Allow and log.")
    created_at = Column(DateTime, default=datetime.utcnow)
    speaker = relationship("Speaker", back_populates="sessions")
    fraud_signals = relationship("FraudSignal", back_populates="session",
                                  cascade="all, delete-orphan")


class FraudSignal(Base):
    __tablename__ = "fraud_signals"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("voice_sessions.id"), nullable=False)
    keyword = Column(String)
    points = Column(Integer)
    category = Column(String)
    session = relationship("VoiceSession", back_populates="fraud_signals")

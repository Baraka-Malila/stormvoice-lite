from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class VoiceSession(Base):
    __tablename__ = "voice_sessions"

    id                 = Column(Integer,  primary_key=True)
    svm_speaker        = Column(String,   nullable=True)
    svm_confidence     = Column(Float,    nullable=True)
    cnn_speaker        = Column(String,   nullable=True)
    cnn_confidence     = Column(Float,    nullable=True)
    transcript         = Column(Text,     nullable=False, default="")
    fraud_category     = Column(String,   nullable=False, default="")
    risk_score         = Column(Integer,  nullable=False, default=0)
    risk_level         = Column(String,   nullable=False, default="Low")
    recommended_action = Column(Text,     nullable=False, default="")
    created_at         = Column(DateTime, default=datetime.utcnow)
    fraud_signals      = relationship("FraudSignal", back_populates="session",
                                      cascade="all, delete-orphan")


class FraudSignal(Base):
    __tablename__ = "fraud_signals"

    id         = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("voice_sessions.id"), nullable=False)
    keyword    = Column(String,  nullable=False)
    points     = Column(Integer, nullable=False)
    category   = Column(String,  nullable=False)
    session    = relationship("VoiceSession", back_populates="fraud_signals")

"""
Database module for Skin Cancer Detection application using SQLAlchemy.
Manages persistent storage of prediction records, model metadata, and lesion image references.
"""

from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionRecord(db.Model):
    """Stores inference history for uploaded skin lesion images."""

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_identifier = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    image_url = db.Column(db.String(512), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # BENIGN / MALIGNANT
    confidence = db.Column(db.Float, nullable=False)       # 0.0 - 100.0
    confidence_percentage = db.Column(db.String(20), nullable=False) # e.g. "92.4%"
    benign_probability = db.Column(db.Float, nullable=False)
    malignant_probability = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), default="Support Vector Machine (SVM)")
    features_summary = db.Column(db.Text, nullable=True)   # JSON string of extracted ABCD features
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert database entity to serializable dictionary."""
        features = {}
        if self.features_summary:
            try:
                features = json.loads(self.features_summary)
            except Exception:
                features = {}

        is_cancer = self.prediction.upper() == "MALIGNANT"
        cancer_status = "SKIN CANCER DETECTED" if is_cancer else "NOT SKIN CANCER"

        return {
            "id": self.id,
            "image_identifier": self.image_identifier,
            "filename": self.filename,
            "image_url": self.image_url,
            "prediction": self.prediction,
            "is_cancer": is_cancer,
            "cancer_status": cancer_status,
            "confidence": round(self.confidence, 1),
            "confidence_percentage": self.confidence_percentage,
            "probabilities": {
                "benign": round(self.benign_probability * 100, 1),
                "malignant": round(self.malignant_probability * 100, 1)
            },
            "model_name": self.model_name,
            "features_summary": features,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        }


def init_db(app):
    """Initializes the database connection and creates tables if not present."""
    db.init_app(app)
    with app.app_context():
        db.create_all()


def save_prediction(
    image_identifier: str,
    filename: str,
    image_url: str,
    prediction: str,
    confidence: float,
    benign_prob: float,
    malignant_prob: float,
    model_name: str,
    features_summary: Optional[Dict[str, Any]] = None
) -> PredictionRecord:
    """Inserts a new skin lesion prediction record into the database."""
    record = PredictionRecord(
        image_identifier=image_identifier,
        filename=filename,
        image_url=image_url,
        prediction=prediction,
        confidence=confidence,
        confidence_percentage=f"{confidence:.1f}%",
        benign_probability=benign_prob,
        malignant_probability=malignant_prob,
        model_name=model_name,
        features_summary=json.dumps(features_summary or {}),
        timestamp=datetime.now(timezone.utc)
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent predictions ordered by timestamp descending."""
    records = PredictionRecord.query.order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
    return [r.to_dict() for r in records]


def delete_record(record_id: int) -> bool:
    """Deletes a single prediction record by ID."""
    record = db.session.get(PredictionRecord, record_id)
    if record:
        db.session.delete(record)
        db.session.commit()
        return True
    return False


def clear_history() -> int:
    """Clears all prediction history from the database."""
    count = PredictionRecord.query.delete()
    db.session.commit()
    return count

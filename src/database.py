"""
Database module for Skin Cancer Detection application using SQLAlchemy.
Manages persistent storage of users, prediction records, model metadata, and lesion image references.
"""

from datetime import datetime, timezone
import json
from typing import List, Dict, Any, Optional
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text

db = SQLAlchemy()


class User(db.Model):
    """User account entity for clinicians, dermatologists, and researchers."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(120), nullable=False, default="Clinician")
    role = db.Column(db.String(80), nullable=False, default="Dermatology Clinician")
    avatar_color = db.Column(db.String(20), default="#0284c7")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship to user's prediction records
    predictions = db.relationship("PredictionRecord", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str):
        """Hashes and stores the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verifies if the provided password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> Dict[str, Any]:
        """Convert user entity to serializable dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "avatar_color": self.avatar_color,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        }


class PredictionRecord(db.Model):
    """Stores inference history for uploaded skin lesion images."""

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
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

        user_name = self.user.full_name if self.user else "Anonymous Clinician"

        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": user_name,
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
    """Initializes database schema, applies migrations if necessary, and seeds default users."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

        # Check and add user_id column to prediction_history if migrating an existing SQLite database
        try:
            db.session.execute(text("ALTER TABLE prediction_history ADD COLUMN user_id INTEGER REFERENCES users(id)"))
            db.session.commit()
        except Exception:
            db.session.rollback()  # Column already exists

        seed_default_users()


def seed_default_users():
    """Seeds default demo accounts for rapid evaluation and testing."""
    if User.query.first() is not None:
        return

    demo_doctor = User(
        username="demo_doctor",
        email="dr.sarah@clinic.med",
        full_name="Dr. Sarah Chen, MD",
        role="Chief Dermatologist / Lead Researcher",
        avatar_color="#0284c7"
    )
    demo_doctor.set_password("password123")

    demo_researcher = User(
        username="researcher",
        email="alex.rivera@research.edu",
        full_name="Alex Rivera, MSc",
        role="Biomedical ML Researcher",
        avatar_color="#10b981"
    )
    demo_researcher.set_password("password123")

    db.session.add(demo_doctor)
    db.session.add(demo_researcher)
    db.session.commit()


def create_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: str = "Dermatology Clinician",
    avatar_color: str = "#0284c7"
) -> User:
    """Creates and persists a new user account."""
    user = User(
        username=username.strip().lower(),
        email=email.strip().lower(),
        full_name=full_name.strip(),
        role=role.strip(),
        avatar_color=avatar_color
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def authenticate_user(username_or_email: str, password: str) -> Optional[User]:
    """Authenticates a user via username or email and password."""
    identifier = username_or_email.strip().lower()
    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if user and user.check_password(password):
        return user
    return None


def get_user_by_id(user_id: int) -> Optional[User]:
    """Fetches user entity by primary key ID."""
    return db.session.get(User, user_id)


def get_user_by_username(username: str) -> Optional[User]:
    """Fetches user entity by username."""
    return User.query.filter_by(username=username.strip().lower()).first()


def get_user_by_email(email: str) -> Optional[User]:
    """Fetches user entity by email."""
    return User.query.filter_by(email=email.strip().lower()).first()


def save_prediction(
    image_identifier: str,
    filename: str,
    image_url: str,
    prediction: str,
    confidence: float,
    benign_prob: float,
    malignant_prob: float,
    model_name: str,
    features_summary: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None
) -> PredictionRecord:
    """Inserts a new skin lesion prediction record into the database."""
    record = PredictionRecord(
        user_id=user_id,
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


def get_history(limit: int = 50, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves the most recent predictions ordered by timestamp descending, optionally filtered by user."""
    query = PredictionRecord.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    records = query.order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
    return [r.to_dict() for r in records]


def delete_record(record_id: int, user_id: Optional[int] = None) -> bool:
    """Deletes a single prediction record by ID, checking user ownership if specified."""
    record = db.session.get(PredictionRecord, record_id)
    if record:
        if user_id and record.user_id and record.user_id != user_id:
            return False
        db.session.delete(record)
        db.session.commit()
        return True
    return False


def clear_history(user_id: Optional[int] = None) -> int:
    """Clears prediction history from the database, optionally filtered by user."""
    query = PredictionRecord.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    count = query.delete()
    db.session.commit()
    return count

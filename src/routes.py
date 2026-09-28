"""
REST API Routes and Controller for Skin Cancer Detection Dashboard.
Handles authentication, image upload, OpenCV preprocessing, SVM model inference, history DB, and sample delivery.
"""

import json
import os
import uuid
from pathlib import Path
import cv2
from flask import (
    Blueprint, request, jsonify, render_template, send_from_directory,
    current_app, session, redirect, url_for
)
from werkzeug.utils import secure_filename

from src.config import Config
from src.preprocessing import ImagePreprocessor
from src.model import SkinCancerModel
from src.database import (
    save_prediction, get_history, delete_record, clear_history,
    create_user, authenticate_user, get_user_by_id, get_user_by_username, get_user_by_email, User
)

api_bp = Blueprint("api", __name__)
preprocessor = ImagePreprocessor(target_size=Config.IMAGE_SIZE)


# ==============================================================================
# Authentication & View Routes
# ==============================================================================

@api_bp.route("/", methods=["GET"])
def index():
    """Renders the main Skin Cancer Detection dashboard."""
    user_id = session.get("user_id")
    is_guest = session.get("is_guest", False) or request.args.get("guest") == "1"

    if request.args.get("guest") == "1":
        session["is_guest"] = True
        is_guest = True

    # If unauthenticated and not in guest mode, redirect to login page (unless in testing mode)
    if not user_id and not is_guest and not current_app.config.get("TESTING"):
        return redirect(url_for("api.login_page"))

    current_user = get_user_by_id(user_id) if user_id else None
    return render_template("index.html", user=current_user, is_guest=is_guest)


@api_bp.route("/login", methods=["GET"])
def login_page():
    """Renders the dedicated login page with 1-click demo access."""
    if session.get("user_id"):
        return redirect(url_for("api.index"))

    logged_out = request.args.get("logged_out") == "1"
    error = request.args.get("error")
    next_url = request.args.get("next") or url_for("api.index")
    return render_template("login.html", logged_out=logged_out, error=error, next=next_url)


@api_bp.route("/register", methods=["GET"])
def register_page():
    """Renders the clinician/researcher account registration page."""
    if session.get("user_id"):
        return redirect(url_for("api.index"))

    error = request.args.get("error")
    return render_template("register.html", error=error)


@api_bp.route("/api/auth/login", methods=["POST"])
def auth_login():
    """
    Handles authentication via JSON or standard form submission.
    Authenticates against database and establishes secure session.
    """
    is_json = request.is_json
    data = request.get_json(silent=True) if is_json else request.form
    if not data:
        data = request.form

    identifier = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password", "")
    remember = data.get("remember") in [True, "true", "1", "on"]
    next_url = request.args.get("next") or data.get("next") or url_for("api.index")

    if not identifier or not password:
        err_msg = "Please provide both username/email and password."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("login.html", error=err_msg, next=next_url), 400

    user = authenticate_user(identifier, password)
    if not user:
        err_msg = "Invalid credentials. Please verify your username/email and password."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 401
        return render_template("login.html", error=err_msg, next=next_url), 401

    # Store user in session
    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    session["user_name"] = user.full_name
    session["user_role"] = user.role
    session["is_guest"] = False
    if remember:
        session.permanent = True

    if is_json:
        return jsonify({
            "success": True,
            "message": f"Welcome back, {user.full_name}!",
            "user": user.to_dict(),
            "redirect_url": next_url
        }), 200

    return redirect(next_url)


@api_bp.route("/api/auth/register", methods=["POST"])
def auth_register():
    """
    Handles new user registration via JSON or form submission.
    Creates user, sets session, and redirects to dashboard.
    """
    is_json = request.is_json
    data = request.get_json(silent=True) if is_json else request.form
    if not data:
        data = request.form

    full_name = data.get("full_name", "").strip()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")
    role = data.get("role", "Dermatology Clinician").strip()

    if not full_name or not username or not email or not password:
        err_msg = "All fields are required."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("register.html", error=err_msg), 400

    if len(password) < 6:
        err_msg = "Password must be at least 6 characters long."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("register.html", error=err_msg), 400

    if confirm_password and password != confirm_password:
        err_msg = "Passwords do not match."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("register.html", error=err_msg), 400

    if get_user_by_username(username):
        err_msg = f"Username '{username}' is already taken."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 409
        return render_template("register.html", error=err_msg), 409

    if get_user_by_email(email):
        err_msg = f"Email '{email}' is already registered."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 409
        return render_template("register.html", error=err_msg), 409

    colors = ["#0284c7", "#0d9488", "#7c3aed", "#2563eb", "#059669", "#d97706"]
    import random
    avatar_color = random.choice(colors)

    user = create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=role,
        avatar_color=avatar_color
    )

    # Establish session
    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    session["user_name"] = user.full_name
    session["user_role"] = user.role
    session["is_guest"] = False

    if is_json:
        return jsonify({
            "success": True,
            "message": "Account created successfully.",
            "user": user.to_dict(),
            "redirect_url": url_for("api.index")
        }), 201

    return redirect(url_for("api.index"))


@api_bp.route("/logout", methods=["GET"])
@api_bp.route("/api/auth/logout", methods=["POST", "GET"])
def logout():
    """Logs out the user, clears session, and redirects to login."""
    session.clear()
    if request.is_json:
        return jsonify({"success": True, "message": "Signed out successfully."}), 200
    return redirect(url_for("api.login_page", logged_out=1))


@api_bp.route("/api/auth/me", methods=["GET"])
def get_current_user():
    """Returns the authenticated user details or unauthenticated status."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({
            "authenticated": False,
            "is_guest": session.get("is_guest", False),
            "user": None
        }), 200

    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "is_guest": False, "user": None}), 200

    return jsonify({
        "authenticated": True,
        "is_guest": False,
        "user": user.to_dict()
    }), 200


# ==============================================================================
# Model, Prediction & Dataset Routes
# ==============================================================================

@api_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint confirming API readiness."""
    return jsonify({
        "status": "healthy",
        "service": "Skin Cancer Detection API",
        "version": "1.0.0"
    }), 200


@api_bp.route("/api/samples", methods=["GET"])
def get_sample_images():
    """Returns the list of curated dermoscopy sample images for rapid testing."""
    samples_file = Path(Config.SAMPLES_FOLDER) / "samples.json"
    if samples_file.exists():
        with open(samples_file, "r") as f:
            samples = json.load(f)
        return jsonify({"success": True, "samples": samples}), 200

    # Fallback to scanning directory
    samples = []
    for f in Path(Config.SAMPLES_FOLDER).glob("*.jpg"):
        samples.append({
            "id": f.name,
            "title": f.stem.replace("_", " ").title(),
            "type": "Benign" if "benign" in f.name.lower() else "Malignant",
            "filename": f.name,
            "relative_url": f"/samples/{f.name}"
        })
    return jsonify({"success": True, "samples": samples}), 200


@api_bp.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Returns verified model architecture and evaluated test performance metrics across all classifiers."""
    try:
        classifier_key = request.args.get("classifier")
        model_service = SkinCancerModel.get_instance()
        metrics = model_service.get_performance_metrics(classifier_key=classifier_key)
        classifiers_list = model_service.get_available_classifiers()
        return jsonify({
            "success": True,
            "model_info": metrics,
            "classifiers": classifiers_list
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve model info: {str(e)}"
        }), 500


@api_bp.route("/api/classifiers", methods=["GET"])
def get_classifiers():
    """Returns list of available trained ML classifiers with accuracy benchmarks."""
    try:
        model_service = SkinCancerModel.get_instance()
        classifiers = model_service.get_available_classifiers()
        return jsonify({
            "success": True,
            "classifiers": classifiers,
            "count": len(classifiers)
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/predict", methods=["POST"])
def predict_lesion():
    """
    Main prediction endpoint.
    Accepts multipart/form-data with either 'image' file upload or 'sample_id' string.
    Executes OpenCV preprocessing, feature extraction, SVM classification, and database logging.
    """
    uploaded_file = request.files.get("image")
    sample_id = request.form.get("sample_id")

    img_bgr = None
    original_filename = ""
    saved_filename = ""

    try:
        if uploaded_file and uploaded_file.filename != "":
            # Validate file extension and size
            raw_filename = secure_filename(uploaded_file.filename) or "lesion.jpg"
            original_filename = raw_filename
            file_bytes = uploaded_file.read()
            file_size = len(file_bytes)

            preprocessor.validate_image_file(
                filename=original_filename,
                file_size=file_size,
                allowed_extensions=Config.ALLOWED_EXTENSIONS,
                max_size_bytes=Config.MAX_CONTENT_LENGTH
            )

            # Decode via OpenCV
            img_bgr = preprocessor.decode_image_bytes(file_bytes)

            # Save uploaded original file
            unique_prefix = uuid.uuid4().hex[:10]
            saved_filename = f"{unique_prefix}_{original_filename}"
            save_path = Path(Config.UPLOADS_FOLDER) / saved_filename
            with open(save_path, "wb") as f:
                f.write(file_bytes)

        elif sample_id:
            # Handle pre-loaded curated sample
            sample_path = Path(Config.SAMPLES_FOLDER) / secure_filename(sample_id)
            if not sample_path.exists():
                return jsonify({"success": False, "error": f"Sample '{sample_id}' not found."}), 404

            original_filename = sample_id
            saved_filename = f"sample_{sample_id}"
            save_path = Path(Config.UPLOADS_FOLDER) / saved_filename

            img_bgr = preprocessor.load_and_validate(str(sample_path))
            cv2.imwrite(str(save_path), img_bgr)

        else:
            return jsonify({
                "success": False,
                "error": "No image provided. Please upload an image file or select a sample."
            }), 400

        # Step 3 & 4: OpenCV Preprocessing & Lesion Segmentation
        preprocessed = preprocessor.process(img_bgr)

        # Save visual overlay with green segmented lesion contour for UI feedback
        overlay_filename = f"contour_{saved_filename}"
        overlay_path = Path(Config.UPLOADS_FOLDER) / overlay_filename
        cv2.imwrite(str(overlay_path), preprocessed["contour_overlay"])

        classifier_key = request.form.get("classifier", "svm").strip().lower()

        # Step 5 & 6: Feature Extraction & Selected Model Prediction + Comparison
        model_service = SkinCancerModel.get_instance()
        prediction_result = model_service.predict(preprocessed, classifier_key=classifier_key)

        user_id = session.get("user_id")

        db_record = save_prediction(
            image_identifier=original_filename,
            filename=saved_filename,
            image_url=f"/uploads/{saved_filename}",
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            benign_prob=prediction_result["probabilities"]["benign"] / 100.0,
            malignant_prob=prediction_result["probabilities"]["malignant"] / 100.0,
            model_name=prediction_result["model_used"],
            features_summary=prediction_result["feature_summary"],
            user_id=user_id
        )

        # Step 7: Return JSON Response
        return jsonify({
            "success": True,
            "prediction": prediction_result["prediction"],
            "is_cancer": prediction_result["is_cancer"],
            "cancer_status": prediction_result["cancer_status"],
            "cancer_verdict": prediction_result["cancer_verdict"],
            "cancer_explanation": prediction_result["cancer_explanation"],
            "action_advice": prediction_result["action_advice"],
            "confidence": prediction_result["confidence"],
            "confidence_percentage": prediction_result["confidence_percentage"],
            "probabilities": prediction_result["probabilities"],
            "is_skin": prediction_result["is_skin"],
            "skin_ratio": prediction_result["skin_ratio"],
            "classifier_key": prediction_result["classifier_key"],
            "model_used": prediction_result["model_used"],
            "model_accuracy": prediction_result["model_accuracy"],
            "kernel": prediction_result["kernel"],
            "image_info": {
                "filename": original_filename,
                "original_url": f"/uploads/{saved_filename}",
                "segmented_url": f"/uploads/{overlay_filename}"
            },
            "feature_summary": prediction_result["feature_summary"],
            "classifiers_comparison": prediction_result["classifiers_comparison"],
            "record_id": db_record.id,
            "timestamp": db_record.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Prediction pipeline error: {str(e)}")
        return jsonify({"success": False, "error": f"Internal prediction failure: {str(e)}"}), 500


@api_bp.route("/api/history", methods=["GET"])
def get_prediction_history():
    """Retrieves recent prediction records from the database, personalized for logged-in user if active."""
    try:
        limit = request.args.get("limit", default=30, type=int)
        user_id = session.get("user_id")
        history = get_history(limit=limit, user_id=user_id)
        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_prediction_record(record_id: int):
    """Deletes a specific prediction record."""
    try:
        user_id = session.get("user_id")
        deleted = delete_record(record_id, user_id=user_id)
        if deleted:
            return jsonify({"success": True, "message": f"Record {record_id} deleted."}), 200
        return jsonify({"success": False, "error": "Record not found or unauthorized."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/history", methods=["DELETE"])
def clear_prediction_history():
    """Clears prediction records."""
    try:
        user_id = session.get("user_id")
        count = clear_history(user_id=user_id)
        return jsonify({"success": True, "message": f"Cleared {count} history records."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Static Asset Delivery Routes
@api_bp.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serves uploaded and processed lesion images."""
    return send_from_directory(Config.UPLOADS_FOLDER, filename)


@api_bp.route("/samples/<path:filename>")
def serve_sample(filename):
    """Serves sample dermoscopy images."""
    return send_from_directory(Config.SAMPLES_FOLDER, filename)

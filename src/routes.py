"""
REST API Routes and Controller for Skin Cancer Detection Dashboard.
Handles image upload, OpenCV preprocessing, SVM model inference, history DB, and sample delivery.
"""

import json
import os
import uuid
from pathlib import Path
import cv2
from flask import Blueprint, request, jsonify, render_template, send_from_directory, current_app
from werkzeug.utils import secure_filename

from src.config import Config
from src.preprocessing import ImagePreprocessor
from src.model import SkinCancerModel
from src.database import save_prediction, get_history, delete_record, clear_history

api_bp = Blueprint("api", __name__)
preprocessor = ImagePreprocessor(target_size=Config.IMAGE_SIZE)


@api_bp.route("/", methods=["GET"])
def index():
    """Renders the main Skin Cancer Detection dashboard."""
    return render_template("index.html")


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
    """Returns verified model architecture and evaluated test performance metrics."""
    try:
        model_service = SkinCancerModel.get_instance()
        metrics = model_service.get_performance_metrics()
        return jsonify({
            "success": True,
            "model_info": metrics
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve model info: {str(e)}"
        }), 500


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

        # Step 5 & 6: Feature Extraction & SVM Model Prediction
        model_service = SkinCancerModel.get_instance()
        prediction_result = model_service.predict(preprocessed)

        # Step 7: Store Prediction History in SQLite Database
        original_url = f"/uploads/{saved_filename}"
        overlay_url = f"/uploads/{overlay_filename}"

        db_record = save_prediction(
            image_identifier=original_filename,
            filename=saved_filename,
            image_url=original_url,
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            benign_prob=prediction_result["probabilities"]["benign"] / 100.0,
            malignant_prob=prediction_result["probabilities"]["malignant"] / 100.0,
            model_name=prediction_result["model_used"],
            features_summary=prediction_result["feature_summary"]
        )

        # Step 8: Return comprehensive JSON response to frontend
        return jsonify({
            "success": True,
            "prediction": prediction_result["prediction"],
            "is_cancer": prediction_result["is_cancer"],
            "cancer_status": prediction_result["cancer_status"],
            "cancer_verdict": prediction_result["cancer_verdict"],
            "cancer_explanation": prediction_result["cancer_explanation"],
            "action_advice": prediction_result["action_advice"],
            "is_skin": prediction_result["is_skin"],
            "skin_ratio": prediction_result["skin_ratio"],
            "model_used": prediction_result["model_used"],
            "kernel": prediction_result["kernel"],
            "confidence": prediction_result["confidence"],
            "confidence_percentage": prediction_result["confidence_percentage"],
            "probabilities": prediction_result["probabilities"],
            "feature_summary": prediction_result["feature_summary"],
            "image_info": {
                "original_url": original_url,
                "segmented_url": overlay_url,
                "filename": original_filename
            },
            "record_id": db_record.id,
            "timestamp": db_record.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        }), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Prediction pipeline error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal processing error: {str(e)}"}), 500


@api_bp.route("/api/history", methods=["GET"])
def get_prediction_history():
    """Retrieves list of previous skin lesion predictions from SQLite database."""
    try:
        limit = request.args.get("limit", default=30, type=int)
        history = get_history(limit=limit)
        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history_record(record_id: int):
    """Deletes an individual history record by ID."""
    try:
        deleted = delete_record(record_id)
        if deleted:
            return jsonify({"success": True, "message": f"Record {record_id} deleted."}), 200
        return jsonify({"success": False, "error": f"Record {record_id} not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/history", methods=["DELETE"])
def clear_all_history():
    """Clears all stored prediction history records."""
    try:
        count = clear_history()
        return jsonify({
            "success": True,
            "message": f"Successfully cleared {count} prediction records."
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename: str):
    """Serves uploaded and preprocessed lesion images."""
    return send_from_directory(Config.UPLOADS_FOLDER, filename)


@api_bp.route("/samples/<path:filename>", methods=["GET"])
def serve_sample(filename: str):
    """Serves curated sample dermoscopy images."""
    return send_from_directory(Config.SAMPLES_FOLDER, filename)

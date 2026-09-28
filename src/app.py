"""
Flask Application Factory for Skin Cancer Detection Dashboard.
Configures REST endpoints, database integration, CORS, error handling, and static assets.
"""

import os
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import Config
from src.database import init_db
from src.routes import api_bp


def create_app(config_class=Config):
    """Factory creating and configuring the Flask application instance."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static"
    )
    app.config.from_object(config_class)

    # Enable CORS for flexible integration
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database
    init_db(app)

    # Register API and view blueprints
    app.register_blueprint(api_bp)

    # Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Bad request", "details": str(error)}), 400
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Endpoint not found"}), 404
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        return jsonify({
            "success": False,
            "error": f"Image file is too large. Maximum allowed size is {mb:.1f} MB."
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)

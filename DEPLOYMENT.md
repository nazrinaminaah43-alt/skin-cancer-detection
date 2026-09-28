# 🚀 Deployment Guide: Skin Cancer Detection Web Application

This document provides step-by-step instructions for deploying the **Skin Cancer Detection Machine Learning Web Application** to production across various platforms.

---

## 📋 Pre-configured Deployment Assets Included in Repository

- **[`Dockerfile`](file:///c:/Users/user/Desktop/AML%20project/Dockerfile)**: Multi-platform container configuration with OpenCV dependencies and Gunicorn WSGI.
- **[`.dockerignore`](file:///c:/Users/user/Desktop/AML%20project/.dockerignore)**: Excludes caches, training dataset, and local DB files from build artifacts.
- **[`Procfile`](file:///c:/Users/user/Desktop/AML%20project/Procfile)**: Process declaration for Heroku, Railway, and Render native web services.
- **[`render.yaml`](file:///c:/Users/user/Desktop/AML%20project/render.yaml)**: Infrastructure-as-code Blueprint for 1-click Render deployment.
- **[`requirements.txt`](file:///c:/Users/user/Desktop/AML%20project/requirements.txt)**: Production dependencies with `opencv-python-headless` and `gunicorn`.

---

## 🌐 Option 1: Deploy on Render (Recommended — 100% Free Web Service)

Render offers free hosting with native Python and Git integration.

### Steps:
1. **Push latest changes to GitHub**:
   ```bash
   git add .
   git commit -m "Add production deployment configurations"
   git push origin main
   ```
2. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
3. Click **New +** > **Web Service**.
4. Select your repository: `nazrinaminaah43-alt/skin-cancer-detection`.
5. Configure the deployment settings:
   - **Name**: `skin-cancer-detection`
   - **Region**: Choose closest to your users (e.g., Oregon, Frankfurt, Singapore)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 run:app`
   - **Plan**: `Free`
6. (Optional) In **Advanced Settings**, add Environment Variables:
   - `SECRET_KEY`: `<generate-a-random-secret-key>`
   - `FLASK_DEBUG`: `False`
7. Click **Create Web Service**.
8. Render will clone the repo, install packages, and deploy your live URL (e.g. `https://skin-cancer-detection.onrender.com`).

---

## 🤗 Option 2: Deploy on Hugging Face Spaces (Free ML Cloud)

Hugging Face Spaces is designed specifically for machine learning applications.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space Name: `skin-cancer-detection`
3. License: `mit` or `open-source`
4. Select **Space SDK**: **Docker** (Blank).
5. Choose Hardware: **CPU Basic (Free - 2 vCPU, 16 GB RAM)**.
6. Click **Create Space**.
7. In your local terminal, add the Hugging Face remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/skin-cancer-detection
   git push hf main
   ```
   *(Or sync automatically via GitHub Actions / Space Settings).*
8. Hugging Face Spaces will automatically build the `Dockerfile` and serve on port `7860`.

---

## 🚂 Option 3: Deploy on Railway

1. Sign in to [railway.app](https://railway.app/).
2. Click **New Project** > **Deploy from GitHub repo**.
3. Select `nazrinaminaah43-alt/skin-cancer-detection`.
4. Railway will automatically detect `Procfile` or `Dockerfile` and deploy the service.
5. In **Settings** > **Networking**, click **Generate Domain** to get a public HTTPS URL.

---

## 🐳 Option 4: Deploy Locally with Docker

If Docker Desktop is installed:

```bash
# 1. Build Docker image
docker build -t skin-cancer-detection .

# 2. Run container on port 5000
docker run -d -p 5000:5000 --name skin-cancer-app skin-cancer-detection

# 3. View running app
# Open http://localhost:5000 in your browser
```

---

## 💻 Option 5: Local Production Server (Windows / macOS / Linux)

To run the application locally in production mode without Docker:

```bash
# Launch server
python run.py
```
Access dashboard at `http://127.0.0.1:5000`.

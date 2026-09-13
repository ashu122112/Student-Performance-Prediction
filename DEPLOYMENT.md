# Deployment Guide: Student Performance Prediction Web Service

This guide explains how the machine learning pipeline and Flask web application are packaged, containerized, and deployed to production.

---

## 📌 Resume Alignment

> **Resume Line:**  
> *"Deployed the selected model via Flask, enabling real-time math-score predictions using a persisted preprocessing and inference pipeline."*

### How each part of that statement is fulfilled:
1. **"Deployed the selected model via Flask"**:  
   `app.py` exposes a Flask application serving HTML/JSON prediction endpoints, configured with WSGI (`gunicorn`) and dynamic port binding for container and cloud environments.
2. **"Real-time math-score predictions"**:  
   The `/predictdata` route receives live user form inputs or API payloads and calculates predictions synchronously with sub-second latency.
3. **"Persisted preprocessing and inference pipeline"**:  
   - Preprocessing (`StandardScaler` + `OneHotEncoder` via `ColumnTransformer`) is persisted in `artifacts/preprocessor.pkl`.
   - Best-performing ML model is persisted in `artifacts/model.pkl`.
   - The `PredictPipeline` class loads these pre-computed artifacts at inference time, guaranteeing **zero data leakage** between training and inference.

---

## 🚀 Option 1: Live Cloud Deployment on Render (100% Free, Recommended)

You can deploy this project live to the internet in 2 minutes:

1. **Push this repository to GitHub**:
   ```bash
   git add .
   git commit -m "Add production deployment configurations (Dockerfile, Procfile, Gunicorn)"
   git push origin main
   ```
2. **Go to [render.com](https://render.com)** and log in with your GitHub account.
3. Click **"New +"** → **"Web Service"**.
4. Select your repository `Student-Performance-Prediction`.
5. Render will automatically detect `render.yaml` or set the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 4 --timeout 120`
6. Click **"Deploy Web Service"**.
7. In ~2 minutes, Render gives you a public URL (e.g. `https://student-performance-prediction.onrender.com`) to put directly on your resume!

---

## 🐳 Option 2: Docker Container Deployment

### Build the Docker Image
```bash
docker build -t student-performance-prediction:latest .
```

### Run the Container
```bash
docker run -d -p 8080:8080 --name ml-app student-performance-prediction:latest
```

Access the application in your browser:
👉 **[http://localhost:8080](http://localhost:8080)**

---

## 💻 Option 3: Local Production Run with Gunicorn

To run the production WSGI server locally:
```bash
# Ensure gunicorn is installed
pip install gunicorn

# Start Gunicorn server with 4 worker processes
gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
```

---

## ☁️ Option 4: AWS Elastic Beanstalk / EC2 Deployment

1. Initialize Elastic Beanstalk CLI:
   ```bash
   eb init -p python-3.9 student-performance-app --region us-east-1
   ```
2. Create environment and deploy:
   ```bash
   eb create student-performance-env
   ```
3. Open live app:
   ```bash
   eb open
   ```

---

## 🧪 Architecture Flow Diagram

```
User Browser (HTTP Request)
       │
       ▼
Gunicorn WSGI Server (4 Workers)
       │
       ▼
Flask Application (app.py)
       │
       ├── CustomData Adapter (Constructs DataFrame from user input)
       │
       ▼
PredictPipeline (src/pipeline/prediction_pipeline.py)
       ├── Load artifacts/preprocessor.pkl ──► Transform features (No leakage)
       ├── Load artifacts/model.pkl        ──► Predict math_score
       │
       ▼
Flask Template (templates/index.html with glassmorphism UI)
       │
       ▼
Real-time Output displayed to user
```

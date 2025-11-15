# EPL Predictor - Render Deployment Guide

## Quick Deploy to Render (Free)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/epl-predictor.git
git push -u origin main
```

### 2. Deploy on Render
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: epl-predictor
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --config gunicorn_config.py app:app`
   - **Instance Type**: Free

### 3. Environment Variables (Optional)
- Add `OPENAI_API_KEY` if you have one for the chatbot

### 4. Your Live URL
After deployment, you'll get a URL like:
**https://epl-predictor-XXXX.onrender.com**

## Alternative: One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/epl-predictor)

## Local Testing
```bash
pip install -r requirements.txt
python app.py
# Visit: http://localhost:5000
```

## Features Available
- ✅ Match Predictions with ML
- ✅ Team Analysis & Comparison  
- ✅ Interactive Dashboard
- ✅ AI Chatbot (with OpenAI key)
- ✅ Live Match Tracking
- ✅ Responsive Design

## Troubleshooting
- If build fails, check requirements.txt
- For memory issues, reduce model complexity
- Check logs in Render dashboard
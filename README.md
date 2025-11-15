# EPL Points Prediction

A complete Machine Learning web application for predicting English Premier League match outcomes and team points using ML and Generative AI.

## Features

- **ScoreSight**: Live and upcoming match predictions
- **AI Predictions**: Interactive match outcome prediction with 21 features
- **Dashboard**: Comprehensive EDA visualizations and statistics
- **Team Analysis**: Compare teams on various performance metrics
- **AI Chatbot**: Gen-AI powered assistant for EPL insights
- **User Authentication**: Login/Register system

## Quick Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Local Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set OpenAI API key: `export OPENAI_API_KEY=your_key_here`
4. Run data processing: `python src/1_data_loading.py`
5. Train models: `python src/3_train_model.py`
6. Start the app: `python app.py`

## Live Demo

🚀 **Live URL**: [https://epl-predictor.onrender.com](https://epl-predictor.onrender.com)

*Note: First load may take 30-60 seconds as the free tier spins up*

## Project Structure

```
EPL_Points_Prediction/
├── data/                     # Raw + processed datasets
├── models/                   # Saved ML models (.joblib)
├── outputs/                  # EDA plots and figures
├── src/                      # Python scripts
├── templates/                # HTML pages
├── static/                   # CSS + JS + images
├── README.md
├── requirements.txt
└── app.py                    # Entry point
```

## Models

- **Model 1**: RandomForestRegressor for total points prediction
- **Model 2**: RandomForestClassifier for match result prediction (H/D/A)
- **Model 3**: Full features classifier using 15 match statistics

## Tech Stack

- **Backend**: Flask, Python
- **ML**: Scikit-learn, Pandas, NumPy
- **Frontend**: Bootstrap 5, HTML/CSS/JS
- **Visualization**: Matplotlib, Seaborn, Plotly
- **AI**: OpenAI GPT-3.5-turbo
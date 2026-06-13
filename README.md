# VALORANT Match Predictor 🎮🤖

A machine learning-powered web application that predicts the outcome of a Valorant match based on your in-game statistics. Developed as a final project for Data Science (UAS DS).

## 🌟 Features

-   **Match Prediction**: Calculates your probability of winning using a Gradient Boosting Classifier model.
-   **Tactical Analyst**: Provides context-aware performance advice (green = strength, red = area for improvement, yellow = neutral) by comparing your input against dataset averages.
-   **Feature Impact Dashboard**: Visualizes the top metrics that influence your match prediction outcome using animated horizontal bars.
-   **What-If Simulator**: An interactive simulator to test how changes in your K/D Ratio and First Bloods affect your win probability in real-time.
-   **Live KPI Counters**: Dynamic animated counters summarizing key dataset metrics.
-   **Prediction History**: Automatically saves your recent predictions locally for quick comparison.
-   **Interactive Map Preview**: Displays the selected Valorant map with a modern HUD-style overlay.
-   **Responsive Design**: A sleek, mobile-friendly interface inspired by Valorant's aesthetic.

## 🛠️ Technology Stack

-   **Backend**: Python, Flask
-   **Machine Learning**: Scikit-Learn (Gradient Boosting Classifier), Pandas, NumPy
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **Deployment**: Ready for Render / Heroku

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anndaanhr/UASDS.git
    cd UASDS
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

4.  **Open your browser:**
    Navigate to `http://127.0.0.1:5000`

## 🧠 Model Training

The prediction engine runs on a pre-trained `GradientBoostingClassifier` saved as `model.pkl`. The model evaluates:
-   Map and Average Rank (Encoded)
-   Econ Rating
-   First Bloods
-   Spike Plants
-   Total Kills & Deaths
-   K/D Ratio

*Model metadata, including feature importance and dataset averages, is stored in `model_metadata.json`.*

## 👨‍💻 Created By

**Ananda Anhar Subing**
UAS Data Science 2026

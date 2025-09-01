# 🛡️ AI-Driven Forensic Platform for Market Fraud Detection

## 📖 About the Project
This project is an **AI-driven fraud detection and alert system** designed to protect retail investors from market manipulation.  
India has over **10 crore retail investors**, making them vulnerable to pump-and-dump schemes, insider trading, and fake news.  
This platform acts as an **early-warning system** by monitoring market data and news to detect suspicious activities and deliver **real-time, explainable risk alerts**.

The solution is designed for scalability and integration with broker platforms, supporting SEBI’s mandate to safeguard investors and ensure transparency.

**Demo Video link:** https://drive.google.com/file/d/1SfB3yTQ5LDk6LdeoappGiofGMA_eCLiH/view?usp=sharing

---

## ✨ Features

### 🔹 Basic Features (MVP)
- **Market Anomaly Detection**: Detects unusual market activity (volume/price spikes, abnormal volatility).
- **Circuit Breaker Monitoring**: Tracks stocks hitting price circuit breakers.
- **News Sentiment Analysis**: Assesses financial news sentiment to gauge market mood.
- **Composite Risk Score**: Generates a Low, Medium, or High-risk score.
- **Streamlit Dashboard**: Intuitive dashboard for risk visualization.
- **Email Alerts**: Sends real-time high-risk stock alerts.

### 🔹 Advanced Features (Stretch Goals)
- **Order-Book Anomaly Detection**: Detects spoofing & manipulation patterns in the order book.
- **Suspicious Trader Clustering**: Identifies clusters of suspicious traders using graph analysis.
- **Social Media Integration**: Incorporates sentiment from social platforms.
- **What-If Simulator**: Projects potential losses for investors.
- **Broker API Integration**: Supports APIs like Zerodha and Groww.

---

## ⚙️ System Architecture

### 📊 Data Ingestion
- OHLCV (Open, High, Low, Close, Volume) market data.
- Curated financial news headlines.

### 🛠️ Feature Engineering
- **Volume Spike Analysis** (Z-score based).
- **Price Gap Analysis** (previous close vs. opening price).
- **Volatility Measurement** (True Range).

### 🧠 Model Layer
- **Isolation Forest** for anomaly detection.
- **XGBoost** for supervised scoring.
- **Transformers (DistilBERT/MiniLM)** for news sentiment analysis.

### 💻 Frontend
- **Streamlit Dashboard** with risk tables, trend charts, and alert logs.

### 🧩 Explainability
- **SHAP Values** for explainability.
- **Human-readable risk alerts**: e.g., "Stock flagged due to 3× volume spike + 2 negative news items."

---

## 🛠️ Tech Stack
- **Language**: Python 3.11
- **Machine Learning**: Pandas, Scikit-learn, XGBoost, Hugging Face Transformers
- **Visualization**: Plotly
- **Frontend**: Streamlit
- **MLOps**: MLflow

---

## 🚀 Getting Started

### 🔑 Prerequisites
- Python 3.11
- Install libraries manually using pip (see below).

### 📦 Installation

```bash
pip install streamlit pandas numpy scikit-learn plotly xgboost shap mlflow
```

---

### ▶️ Running the App

```bash
# Run the Streamlit dashboard
streamlit run app_streamlit.py
```

If no `alerts.csv` is uploaded, the app will load a demo dataset.

---

## 📂 Project Structure

```
📦 AI-Driven-Fraud-Detection
 ┣ 📜 app_streamlit.py       # Main Streamlit dashboard
 ┣ 📜 README.md              # Documentation
 ┣ 📜 alerts.csv (optional)  # Market data input
```

---

## 🤝 Contributing
We welcome contributions!  
- Fork the repository.
- Create a feature branch.
- Submit a pull request.

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
For inquiries, contact the project owner.
K.Chaitanya Sai (Main Lead) 
K. Sai Shashank (Team Member)
---

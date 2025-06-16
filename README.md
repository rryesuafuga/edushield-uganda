
# EduShield UG: School Dropout Risk Predictor & Intervention System

EduShield UG is a machine learning-powered application designed to identify Ugandan students at risk of dropping out of school and recommend targeted interventions. This project leverages national survey data to support proactive education policy and school-level decision-making.

## 🔍 Project Objective

- Predict dropout risk using socioeconomic, demographic, and educational factors.
- Recommend evidence-based interventions (e.g., school feeding, counseling).
- Provide regional analytics and early warning alerts for policymakers.

## 📁 Project Structure

```
edushield-uganda/
│
├── data/
│   ├── raw/                # Original CSVs (not tracked in Git)
│   └── processed/          # Cleaned datasets ready for modeling
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── app/
│   ├── dashboard.py        # Web interface (e.g., Streamlit or Flask)
│   └── recommender.py      # Intervention logic
│
├── models/
│   └── dropout_predictor.pkl
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore unneeded files
└── README.md               # Project overview
```

## 🧪 Data Sources

- Uganda National Household Survey (UNHS) 2019/20
- Uganda Demographic and Health Survey (UDHS) 2022
- Uganda National Panel Survey (UNPS) 2019/20
- UNESCO Institute for Statistics (UIS) – Education indicators

## 🚀 Tech Stack

- Python (pandas, scikit-learn, XGBoost)
- Jupyter Notebooks
- Streamlit or Flask (for the dashboard)
- PostgreSQL or SQLite
- GitHub + Streamlit Cloud/Render for deployment

## 📌 Disclaimer

This tool is for **educational and policy support purposes** only. It does not store or expose personally identifiable data.

## 👤 Author

Raymond Wayesu – Biostatistician & Data Scientist

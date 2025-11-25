# Black Friday Sales Analysis - Big Data Project

**Projekt Big Data z wykorzystaniem Apache Spark, Kafka, i MLlib**

## 📋 Opis Projektu

Kompleksowy system analityczny do analizy i predykcji zachowań zakupowych klientów podczas Black Friday, wykorzystujący technologie Big Data.

### Technologie:
- **Apache Spark 3.5+** - Distributed data processing
- **Apache Kafka** - Real-time streaming
- **PySpark MLlib** - Machine Learning
- **Delta Lake** - Lakehouse architecture
- **Python 3.11+** - Development

## 🎯 Cele Projektu

1. **Predykcja wartości zakupów** - Regression models (Linear, Random Forest, GBT)
2. **Segmentacja klientów** - K-Means clustering
3. **System rekomendacji** - Collaborative Filtering (ALS)
4. **Real-time analytics** - Spark Streaming + Kafka

## 📁 Struktura Projektu

```
BlackFriday/
├── config/              # Spark & Kafka configuration
│   └── spark_config.py
├── data/
│   ├── raw/            # Raw CSV files (download from Kaggle)
│   ├── processed/      # Processed data (Delta Lake)
│   └── streaming/      # Streaming data simulation
├── docs/               # Project documentation
│   └── ZALOZENIA_PROJEKTOWE_BLACK_FRIDAY.md
├── notebooks/          # Jupyter notebooks
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── src/                # Source code
├── models/             # Trained ML models
├── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/vladyslavusatenko/BigDataSpark.git
cd BigDataSpark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download Black Friday dataset from [Kaggle](https://www.kaggle.com/datasets/sdolezel/black-friday):
- Place `train.csv` in `data/raw/`
- Place `test.csv` in `data/raw/`

### 3. Windows Setup (Hadoop binaries)

```bash
# Run setup script for Windows
python setup_windows_spark.py
```

### 4. Run Notebooks

```bash
jupyter notebook notebooks/
```

Start with:
1. `01_eda.ipynb` - Exploratory Data Analysis
2. `02_feature_engineering.ipynb` - Feature Engineering Pipeline

## 📊 Features

### Feature Engineering
- **User-level aggregations** (purchase patterns, RFM)
- **Product-level features** (popularity, pricing)
- **Category aggregations**
- **Interaction features**
- **Categorical encoding** (StringIndexer, OneHotEncoder)

### Machine Learning Models
1. **Regression** - Purchase prediction (RMSE, MAE, R²)
2. **Clustering** - Customer segmentation (Silhouette Score)
3. **Recommendation** - ALS collaborative filtering (Precision@K)

### Streaming Pipeline
- Apache Kafka producer/consumer
- Spark Structured Streaming
- Real-time aggregations
- Delta Lake integration

## 📈 Results

Expected outcomes:
- Purchase prediction accuracy: R² > 0.85
- Customer segments: 4-6 distinct groups
- Recommendation system: Precision@10 > 0.3

## 📚 Documentation

Full project documentation (in Polish):
- [Założenia Projektowe](docs/ZALOZENIA_PROJEKTOWE_BLACK_FRIDAY.md)

## 🛠️ Tech Stack

```
Data Processing:
├── Apache Spark 3.5.x
├── Apache Kafka 3.6.x
├── Delta Lake 3.0.x
└── Hadoop 3.3.x

Machine Learning:
├── Spark MLlib
└── Scikit-learn

Visualization:
├── Matplotlib
├── Seaborn
└── Plotly

Development:
└── Jupyter Notebook
```

## 👨‍💻 Author

**Vlad Usatenko**
- University: Politechnika Łódzka
- Course: Big Data
- Year: 2025

## 📝 License

This project is created for educational purposes.

Dataset: [Kaggle Black Friday Dataset](https://www.kaggle.com/datasets/sdolezel/black-friday) (ODbL License)

## 🔗 Links

- GitHub Repository: https://github.com/vladyslavusatenko/BigDataSpark
- Kaggle Dataset: https://www.kaggle.com/datasets/sdolezel/black-friday
- Apache Spark: https://spark.apache.org/
- Apache Kafka: https://kafka.apache.org/

---

**Status:** 🚧 In Development

**Last Updated:** November 25, 2025

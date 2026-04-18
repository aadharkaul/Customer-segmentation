# 🛍️ Customer Segmentation with RFM Analysis & K-Means Clustering

> **Identifying high-value customer groups to drive targeted marketing strategy**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Business Problem

Not all customers are equal. A small segment drives the majority of revenue — but most marketing budgets treat everyone the same.

This project uses **RFM (Recency, Frequency, Monetary) analysis** combined with **K-Means clustering** to segment ~5,800 customers into distinct behavioral groups, each with tailored marketing recommendations.

---

## 📁 Project Structure

```
customer-segmentation/
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_rfm_analysis.ipynb  # RFM scoring
│   ├── 03_clustering.ipynb    # K-Means + evaluation
│   └── 04_insights.ipynb      # Segment profiles & strategy
├── src/
│   ├── rfm.py                 # Reusable RFM module
│   └── clustering.py          # Clustering pipeline
├── data/
│   └── online_retail_II.xlsx  # Dataset (download separately)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Online Retail II** — UCI Machine Learning Repository  
🔗 https://archive.ics.uci.edu/dataset/502/online+retail+ii

- ~1 million transactions
- UK-based online retailer, 2009–2011
- Contains: Invoice, Product, Quantity, Price, Customer ID, Country

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation
cd customer-segmentation
pip install -r requirements.txt

# Download dataset from UCI and place in data/ folder
# Then run notebooks in order: 01 → 02 → 03 → 04
jupyter notebook
```

---

## 🔍 Methodology

```
Raw Data → Clean → RFM Scoring → Normalize → K-Means → Segment Profiles
```

1. **EDA** — understand data quality, sales trends, customer distributions
2. **RFM Scoring** — rank each customer on Recency, Frequency, Monetary value
3. **Clustering** — find optimal k with Elbow + Silhouette, apply K-Means
4. **Insights** — name each segment, profile it, recommend marketing actions

---

## 📈 Key Results

*(To be updated after analysis)*

| Segment | Size | Avg Revenue | Strategy |
|---------|------|-------------|----------|
| Champions | TBD | TBD | Reward & retain |
| Loyal Customers | TBD | TBD | Upsell |
| At Risk | TBD | TBD | Win-back campaign |
| Lost | TBD | TBD | Re-engagement or sunset |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **ML** — scikit-learn (K-Means, StandardScaler)
- **Viz** — matplotlib, seaborn, plotly
- **Notebooks** — Jupyter

---

## 👤 Author

Built as part of a Marketing & Product Data Science portfolio.

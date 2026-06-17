# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

Analyzing e-commerce transaction data to uncover customer purchasing patterns, segment customers using RFM analysis, and recommend products through collaborative filtering — all wrapped in an interactive Streamlit app.

## 📣 Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behavior. Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth.

This project examines transaction data from an online retail business to uncover patterns in customer purchase behavior, segment customers based on **Recency, Frequency, and Monetary (RFM) analysis**, and develop a product **recommendation system** using **collaborative filtering** techniques.

## 📌 Real-Time Business Use Cases

- Customer segmentation for targeted marketing campaigns
- Personalized product recommendations on e-commerce platforms
- Identifying at-risk customers for retention programs
- Dynamic pricing strategies based on purchase behavior
- Inventory management and stock optimization based on customer demand patterns

## 🧠 Problem Type

- Unsupervised Machine Learning – Clustering
- Collaborative Filtering – Recommendation System

## 🗂 Domain

E-Commerce and Retail Analytics

## 📌 Dataset Description

The dataset contains online retail transaction records with the following columns:

| Column | Description |
|---|---|
| `InvoiceNo` | Transaction number |
| `StockCode` | Unique product/item code |
| `Description` | Name of the product |
| `Quantity` | Number of products purchased |
| `InvoiceDate` | Date and time of transaction (2022–2023) |
| `UnitPrice` | Price per product |
| `CustomerID` | Unique identifier for each customer |
| `Country` | Country where the customer is based |

## 🔧 Project Workflow

### Step 1: Dataset Collection and Understanding
- Explore the dataset to understand its structure and data types
- Identify missing values, duplicates, and unusual records

### Step 2: Data Preprocessing
- Remove rows with missing `CustomerID`
- Exclude cancelled invoices (`InvoiceNo` starting with `C`)
- Remove negative or zero quantities and prices

### Step 3: Exploratory Data Analysis (EDA)
- Transaction volume by country
- Top-selling products
- Purchase trends over time
- Monetary distribution per transaction and customer
- RFM distributions
- Elbow curve for cluster selection
- Customer cluster profiles
- Product recommendation heatmap / similarity matrix

### Step 4: Clustering Methodology

1. **Feature Engineering**
   - **Recency** = Latest purchase date in dataset − customer's last purchase date
   - **Frequency** = Number of transactions per customer
   - **Monetary** = Total amount spent by customer
2. Standardize/normalize the RFM values
3. Choose a clustering algorithm (KMeans, DBSCAN, Hierarchical, etc.)
4. Use the Elbow Method and Silhouette Score to decide the number of clusters
5. Run clustering
6. Visualize clusters using a scatter plot or 3D plot of RFM scores
7. Save the best-performing model for Streamlit usage

**Cluster labeling** is based on interpreting RFM averages:

| Characteristics | Segment Label |
|---|---|
| High Recency, High Frequency, High Monetary | **High-Value** |
| Medium Frequency, Medium Monetary | **Regular** |
| Low Frequency, Low Monetary, older Recency | **Occasional** |
| High Recency, Low Frequency, Low Monetary | **At-Risk** |

### Recommendation System Approach
- Item-based collaborative filtering
- Cosine similarity (or another similarity metric) between products based on purchase history (CustomerID–Description matrix)
- Returns the top 5 similar products to an entered product name

## 📱 Streamlit App Features

### 🎯 Product Recommendation Module
Given a product name, the app recommends 5 similar products based on collaborative filtering.

- Text input box for product name
- "Get Recommendations" button
- Displays 5 recommended products as a styled list/card view

### 🎯 Customer Segmentation Module
Given RFM values, the app predicts which customer segment a user belongs to.

- Number inputs for Recency (days), Frequency (number of purchases), and Monetary (total spend)
- "Predict Cluster" button
- Displays the predicted cluster label (High-Value, Regular, Occasional, or At-Risk)

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `StandardScaler` · `KMeans Clustering` · `Cosine Similarity` · `Streamlit` · `Matplotlib` / `Seaborn`

**Topics:** Data Cleaning, Feature Engineering, EDA, RFM Analysis, Customer Segmentation, Collaborative Filtering, Product Recommendation, Machine Learning, Data Visualization, Pivot Tables, Data Transformation, Real-Time Prediction

## 📁 Repository Structure

```
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks (EDA, clustering, recommendation)
├── models/                 # Saved/serialized model files
├── app.py                  # Streamlit application
├── requirements.txt        # Project dependencies
└── README.md
```

> Update this structure to match your actual repository layout.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
streamlit run app.py
```

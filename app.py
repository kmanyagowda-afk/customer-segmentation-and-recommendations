%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(
    page_title="Customer Segmentation & Recommendation System",
    layout="wide"
)

st.title(" Customer Segmentation and Product Recommendation System")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="latin1")

    df.dropna(subset=["CustomerID","Description"], inplace=True)

    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df

df = load_data()

# ---------------- SIDEBAR ----------------

menu = st.sidebar.radio(
    "Select Module",
    ["Dashboard",
     "Customer Segmentation",
     "Recommendation"]
)

# ===================================================
# DASHBOARD
# ===================================================

if menu == "Dashboard":

    st.header(" Dashboard")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        int(df["CustomerID"].nunique())
    )

    col2.metric(
        "Total Products",
        int(df["Description"].nunique())
    )

    col3.metric(
        "Total Revenue",
        f"${df['TotalPrice'].sum():,.0f}"
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Top Selling Products")

    top_products = (
        df["Description"]
        .value_counts()
        .head(10)
    )

    fig = px.bar(
        x=top_products.values,
        y=top_products.index,
        orientation="h",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ===================================================
# CUSTOMER SEGMENTATION
# ===================================================

elif menu == "Customer Segmentation":

    st.header(" Customer Segmentation")

    snapshot_date = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg({
        "InvoiceNo":"count",
        "TotalPrice":"sum"
    })

    rfm.columns = [
        "Frequency",
        "Monetary"
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(rfm)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(X)

    cluster_names = {
        0:"High Value",
        1:"Regular",
        2:"Occasional",
        3:"At Risk"
    }

    st.subheader("Enter Customer Values")

    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=5
    )

    monetary = st.number_input(
        "Monetary Value",
        min_value=0.0,
        value=100.0
    )

    if st.button("Predict Segment"):

        sample = scaler.transform(
            [[frequency, monetary]]
        )

        cluster = kmeans.predict(sample)[0]

        segment_names = {
        0: "High Value Customer",
        1: "Occasional Shopper",
        2: "Regular Shopper",
        3: "At Risk Customer"
         }
        st.success(
        f"This customer belongs to: {segment_names[cluster]}"
          )


    st.subheader("Cluster Distribution")

    cluster_fig = px.pie(
        names=rfm["Cluster"]
    )

    st.plotly_chart(
        cluster_fig,
        use_container_width=True
    )

# ===================================================
# RECOMMENDATION SYSTEM
# ===================================================

elif menu == "Recommendation":

    st.header(" Product Recommendation")

    customer_product = pd.crosstab(
        df["CustomerID"],
        df["Description"]
    )

    similarity = cosine_similarity(
        customer_product.T
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=customer_product.columns,
        columns=customer_product.columns
    )

    product = st.selectbox(
        "Select Product",
        sorted(similarity_df.columns)
    )

    if st.button("Get Recommendations"):

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.success(
            f"Top 5 Similar Products for {product}"
        )

        for i,item in enumerate(
            recommendations.index,
            start=1
        ):
            st.write(
                f"{i}. {item}"
            )

        chart_df = pd.DataFrame({
            "Product":
            recommendations.index,

            "Similarity":
            recommendations.values
        })

        st.subheader(
            "Recommendation Scores"
        )

        st.bar_chart(
            chart_df.set_index(
                "Product"
            )
        )
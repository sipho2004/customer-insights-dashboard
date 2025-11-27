"""
Customer Insights Dashboard App (Streamlit)

Features:
- Upload customer CSV or use sample data
- RFM segmentation (Recency, Frequency, Monetary)
- CLV calculation and segmentation
- Purchase trends visualization
- Demographics analysis (age, gender, location)
- Downloadable reports for all insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title='Customer Insights Dashboard', layout='wide')
st.title('📊 Customer Insights Dashboard')

# ---------------- Sidebar ----------------
st.sidebar.header('Data Options')
use_sample = st.sidebar.checkbox('Use sample data', value=True)
uploaded_file = st.sidebar.file_uploader('Upload customer CSV', type=['csv'])

# ---------------- Sample Data ----------------
def sample_customer_data():
    data = {
        'customer_id': range(1,21),
        'recency_days': np.random.randint(1,100,20),
        'frequency': np.random.randint(1,10,20),
        'monetary': np.random.randint(50,500,20),
        'age': np.random.randint(18,60,20),
        'gender': np.random.choice(['M','F'],20),
        'location': np.random.choice(['Cape Town','Johannesburg','Durban'],20)
    }
    return pd.DataFrame(data)

# ---------------- Load Data ----------------
if use_sample:
    df = sample_customer_data()
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info('Please upload a CSV or use sample data')
        df = None

# ---------------- Dashboard ----------------
if df is not None:
    st.subheader('Customer Data Preview')
    st.dataframe(df)

    # ---------- RFM Segmentation ----------
    st.subheader('RFM Segmentation (KMeans)')
    k = st.slider('Number of RFM segments', 2, 5, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['RFM_Segment'] = kmeans.fit_predict(df[['recency_days','frequency','monetary']])
    st.dataframe(df[['customer_id','RFM_Segment']])

    # ---------- CLV Calculation ----------
    st.subheader('Customer Lifetime Value (CLV)')
    # Simple CLV formula: monetary * frequency * lifespan (assume 3 years) * profit margin 0.3
    df['CLV'] = (df['monetary'] * df['frequency'] * 3 * 0.3).round(2)
    high = df['CLV'].quantile(0.75)
    low = df['CLV'].quantile(0.25)
    def clv_segment(clv):
        if clv >= high:
            return 'High Value'
        elif clv <= low:
            return 'Low Value'
        else:
            return 'Medium Value'
    df['CLV_Segment'] = df['CLV'].apply(clv_segment)
    st.dataframe(df[['customer_id','CLV','CLV_Segment']])

    # ---------- Purchase Trends ----------
    st.subheader('Purchase Trends (Sample Simulation)')
    df['purchase_value'] = df['monetary'] * df['frequency']
    trend_df = df.groupby('RFM_Segment')['purchase_value'].sum().reset_index()
    fig, ax = plt.subplots()
    ax.bar(trend_df['RFM_Segment'], trend_df['purchase_value'], color='skyblue')
    ax.set_xlabel('RFM Segment')
    ax.set_ylabel('Total Purchase Value')
    ax.set_title('Purchase Value by RFM Segment')
    st.pyplot(fig)

    # ---------- Demographics ----------
    st.subheader('Customer Demographics')
    # Gender distribution
    gender_counts = df['gender'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    ax.set_title('Gender Distribution')
    st.pyplot(fig)
    # Location distribution
    loc_counts = df['location'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(loc_counts.index, loc_counts.values, color='lightgreen')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Location Distribution')
    st.pyplot(fig)
    # Age distribution
    fig, ax = plt.subplots()
    ax.hist(df['age'], bins=10, color='orange', edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Age Distribution')
    st.pyplot(fig)

    # ---------- Download Reports ----------
    st.subheader('Download Insights Report')
    csv_buf = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Customer Insights CSV', data=csv_buf, file_name='customer_insights.csv', mime='text/csv')

st.markdown('---')
st.caption('This app provides customer segmentation, CLV calculation, purchase trends, and demographics insights for marketing analysis.')

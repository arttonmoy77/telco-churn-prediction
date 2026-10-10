"""Streamlit dashboard for the Telco churn project.

Run:  streamlit run app.py
"""

import streamlit as st

from churn_model import load_data, train_and_evaluate

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telecom Customer Churn Dashboard")
st.caption("Explore churn patterns in the Telco Customer Churn dataset.")


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_model_results(data):
    return train_and_evaluate(data)


try:
    df = get_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

# Summary metrics
customer_count = len(df)
churn_rate = (df["Churn"] == "Yes").mean() * 100
average_tenure = df["tenure"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Customers", f"{customer_count:,}")
col2.metric("Churn rate", f"{churn_rate:.1f}%")
col3.metric("Average tenure", f"{average_tenure:.1f} months")

# Churn counts
st.subheader("Customers by churn status")
st.bar_chart(df["Churn"].value_counts().rename_axis("Churn"))

# Churn rate by contract
st.subheader("Churn rate by contract type")
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .rename("Churn rate (%)")
)
st.bar_chart(contract_churn)

st.caption(
    "These are patterns in this dataset; they do not show that a feature causes churn."
)

# Model comparison
st.subheader("Model comparison")
with st.spinner("Training and evaluating models..."):
    model_results = get_model_results(df)

st.dataframe(model_results.round(3), width="stretch")

st.caption(
    "Recall shows the share of churners identified. "
    "Average precision summarizes precision and recall across thresholds."
)

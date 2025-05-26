
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Market Trend Predictor", layout="wide")

st.title("Market Trend Predictor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df.head())

    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        model_choice = st.selectbox("Choose prediction model", ["ARIMA", "Random Forest"])

        if model_choice == "ARIMA":
            st.subheader("ARIMA Model")

            order = st.text_input("Enter ARIMA order (p,d,q)", "5,1,0")
            try:
                p, d, q = map(int, order.split(","))
                model = sm.tsa.ARIMA(df['Close'], order=(p, d, q))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=30)

                st.line_chart(pd.concat([df['Close'], forecast.rename("Forecast")]))
            except Exception as e:
                st.error(f"Error fitting ARIMA model: {e}")

        elif model_choice == "Random Forest":
            st.subheader("Random Forest Regressor")

            df['Day'] = df.index.dayofyear
            X = df[['Day']]
            y = df['Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)

            st.line_chart(result_df)
            st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
else:
    st.info("Please upload a CSV file with 'Date' and 'Close' columns.")

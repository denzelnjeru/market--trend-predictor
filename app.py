
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Market Trend Predictor", layout="wide")
st.title("Market Trend Predictor")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

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

        model_choice = st.selectbox("Choose prediction model", ["ARIMA", "Random Forest", "LSTM"])

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

        elif model_choice == "LSTM":
            st.subheader("LSTM Model (PyTorch)")
            close_prices = df['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_prices)
            sequence_length = 30
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:i+sequence_length])
                y.append(scaled_data[i+sequence_length])
            X = np.array(X)
            y = np.array(y)

            X_train = torch.tensor(X[:-30], dtype=torch.float32)
            y_train = torch.tensor(y[:-30], dtype=torch.float32)
            X_test = torch.tensor(X[-30:], dtype=torch.float32)

            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(100):
                model.train()
                output = model(X_train)
                loss = criterion(output, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(X_test).numpy()
            predictions = scaler.inverse_transform(predictions)
            actual = df['Close'][-30:].values

            result_df = pd.DataFrame({
                'Actual': actual.flatten(),
                'Predicted': predictions.flatten()
            }, index=df.index[-30:])
            st.line_chart(result_df)
else:
    st.info("Please upload a CSV file with 'Date' and 'Close' columns.")

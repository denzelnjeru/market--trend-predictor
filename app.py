#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


# In[ ]:


# Load and preprocess data
df = pd.read_csv('retail_sales_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
target_variable = 'Total Amount'

# Generate synthetic sentiment
def generate_review(row):
    if row['Total Amount'] > 1000:
        return f"I loved this {row['Product Category']} purchase! Great quality!"
    elif row['Total Amount'] > 500:
        return f"Pretty good {row['Product Category']} item."
    else:
        return f"Disappointing {row['Product Category']} experience."

sid = SentimentIntensityAnalyzer()
df['Review'] = df.apply(generate_review, axis=1)
df['Sentiment'] = df['Review'].apply(lambda x: sid.polarity_scores(x)['compound'])


# In[ ]:


# Train-test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Data Scaling
scaler_no_sent = MinMaxScaler()
scaler_with_sent = MinMaxScaler()
train_scaled_no_sent = scaler_no_sent.fit_transform(train[[target_variable]])
test_scaled_no_sent = scaler_no_sent.transform(test[[target_variable]])
train_scaled_with_sent = scaler_with_sent.fit_transform(train[[target_variable, 'Sentiment']])
test_scaled_with_sent = scaler_with_sent.transform(test[[target_variable, 'Sentiment']])

# Sequence creation for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 14
X_train_no_sent, y_train_no_sent = create_sequences(train_scaled_no_sent, seq_length)
X_test_no_sent, y_test_no_sent = create_sequences(test_scaled_no_sent, seq_length)
X_train_with_sent, y_train_with_sent = create_sequences(train_scaled_with_sent, seq_length)
X_test_with_sent, y_test_with_sent = create_sequences(test_scaled_with_sent, seq_length)


# In[ ]:


# Improved LSTM model
def build_improved_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM models with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_no_sent = build_improved_lstm_model((seq_length, 1))
lstm_no_sent.fit(X_train_no_sent, y_train_no_sent, validation_split=0.2,
                 epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)
lstm_pred_no_sent_scaled = lstm_no_sent.predict(X_test_no_sent, verbose=0)

lstm_with_sent = build_improved_lstm_model((seq_length, 2))
lstm_with_sent.fit(X_train_with_sent, y_train_with_sent, validation_split=0.2,
                   epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)
lstm_pred_with_sent_scaled = lstm_with_sent.predict(X_test_with_sent, verbose=0)


# In[ ]:


# Rescale predictions and targets for LSTM
lstm_pred_no_sent = scaler_no_sent.inverse_transform(lstm_pred_no_sent_scaled)
y_test_no_sent_rescaled = scaler_no_sent.inverse_transform(y_test_no_sent.reshape(-1, 1))

dummy_sentiment = np.zeros_like(lstm_pred_with_sent_scaled)
lstm_pred_with_sent = scaler_with_sent.inverse_transform(
    np.concatenate((lstm_pred_with_sent_scaled, dummy_sentiment), axis=1)
)[:, 0].reshape(-1, 1)

y_test_with_sent_rescaled = scaler_with_sent.inverse_transform(
    np.concatenate((y_test_with_sent.reshape(-1, 1), dummy_sentiment), axis=1)
)[:, 0].reshape(-1, 1)

# Trim test sets for LSTM alignment
adjusted_test_no_sent = test.iloc[seq_length:]
adjusted_test_with_sent = test.iloc[seq_length:]


# In[ ]:


# Prepare Random Forest data
def prepare_rf_data(df, include_sentiment=False):
    df_rf = df.copy()
    df_rf['Day'] = df_rf.index.day
    df_rf['Month'] = df_rf.index.month
    df_rf['Weekday'] = df_rf.index.weekday
    df_rf = pd.get_dummies(df_rf, columns=['Gender', 'Product Category'])
    y = df_rf[target_variable]
    drop_cols = ['Transaction ID', 'Customer ID', 'Review', target_variable]
    if not include_sentiment:
        drop_cols.append('Sentiment')
    X = df_rf.drop(columns=drop_cols)
    return X, y

X_train_rf_no_sent, y_train_rf_no_sent = prepare_rf_data(train, include_sentiment=False)
X_test_rf_no_sent, y_test_rf_no_sent = prepare_rf_data(test, include_sentiment=False)
X_test_rf_no_sent = X_test_rf_no_sent.reindex(columns=X_train_rf_no_sent.columns, fill_value=0)

rf_no_sent = RandomForestRegressor(n_estimators=100, random_state=42)
rf_no_sent.fit(X_train_rf_no_sent, y_train_rf_no_sent)
rf_pred_no_sent = rf_no_sent.predict(X_test_rf_no_sent)

X_train_rf_with_sent, y_train_rf_with_sent = prepare_rf_data(train, include_sentiment=True)
X_test_rf_with_sent, y_test_rf_with_sent = prepare_rf_data(test, include_sentiment=True)
X_test_rf_with_sent = X_test_rf_with_sent.reindex(columns=X_train_rf_with_sent.columns, fill_value=0)

rf_with_sent = RandomForestRegressor(n_estimators=100, random_state=42)
rf_with_sent.fit(X_train_rf_with_sent, y_train_rf_with_sent)
rf_pred_with_sent = rf_with_sent.predict(X_test_rf_with_sent)


# In[ ]:


# ARIMA models
sales_daily = df[[target_variable, 'Sentiment']].resample('D').agg({target_variable: 'sum', 'Sentiment': 'mean'})
sales_daily['Sentiment'] = sales_daily['Sentiment'].fillna(0)
train_size_arima = int(len(sales_daily) * 0.8)
train_arima = sales_daily.iloc[:train_size_arima]
test_arima = sales_daily.iloc[train_size_arima:]

arima_no_sent = ARIMA(train_arima[target_variable], order=(5, 1, 0))
arima_no_sent_fit = arima_no_sent.fit()
arima_pred_no_sent = arima_no_sent_fit.forecast(steps=len(test_arima))

arima_with_sent = ARIMA(train_arima[target_variable], order=(5, 1, 0), exog=train_arima['Sentiment'])
arima_with_sent_fit = arima_with_sent.fit()
arima_pred_with_sent = arima_with_sent_fit.forecast(steps=len(test_arima), exog=test_arima['Sentiment'])


# In[ ]:


# Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"{name}:")
    print("  RMSE:", round(sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("  MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("  R²  :", round(r2_score(y_true, y_pred), 2))
    print()

print("\n✅ Final Model Evaluation\n")
evaluate_model("LSTM (No Sentiment)", adjusted_test_no_sent[[target_variable]].values, lstm_pred_no_sent)
evaluate_model("LSTM (With Sentiment)", adjusted_test_with_sent[[target_variable]].values, lstm_pred_with_sent)
evaluate_model("Random Forest (No Sentiment)", y_test_rf_no_sent, rf_pred_no_sent)
evaluate_model("Random Forest (With Sentiment)", y_test_rf_with_sent, rf_pred_with_sent)
evaluate_model("ARIMA (No Sentiment)", test_arima[target_variable].values, arima_pred_no_sent)
evaluate_model("ARIMA (With Sentiment)", test_arima[target_variable].values, arima_pred_with_sent)


# In[ ]:


# Plot ARIMA predictions
plt.figure(figsize=(14, 6))
plt.plot(test_arima.index, test_arima[target_variable], label='Actual Sales')
plt.plot(test_arima.index, arima_pred_no_sent, label='ARIMA No Sentiment')
plt.plot(test_arima.index, arima_pred_with_sent, label='ARIMA With Sentiment')
plt.title('ARIMA Prediction vs Actual')
plt.legend()
plt.grid(True)
plt.show()


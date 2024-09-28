import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import os

# Load the pre-trained model
model = load_model('stock.keras')

# Function to get windowed data
def get_windowed(data, n):
    df = dc(data)
    for i in range(1, n+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# Set backcandle value
backcandle = 10

# Streamlit app
st.title('Stock Price Prediction App using LSTM')

# Upload CSV file
stock = st.text_input('Enter stock symbol:', 'ADANIENT', key='stock', disabled=True)
file = f"dataset/{stock}.csv"

if file is not None:
    data = pd.read_csv(file)

    st.subheader('Raw Data')
    st.write(data.tail())

    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    # Plot raw data
    st.subheader('Closing Price Over Time')
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'])
    st.pyplot(fig)

    # Prepare data for prediction
    temp = data[-(backcandle+1):]
    window_df = get_windowed(temp, backcandle)
    np_df = window_df.to_numpy()
    dates = np_df[:, 0]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_df = scaler.fit_transform(np_df[:, 1:])

    X = scaled_df[:, 1:]
    y = scaled_df[:, 0]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Flip X to match model's expected input
    X = dc(np.flip(X, axis=1))

    # Make prediction
    prediction = model.predict(X).flatten()
    prediction = prediction[-1]
    # Inverse transform the prediction
    prediction = scaler.inverse_transform([[prediction] + [0]*(backcandle)])[:, 0][0]

    st.subheader('Predicted Closing Price for Next Day')
    st.write(f"${prediction:.2f}")

    # Append prediction to data for plotting
    last_date = data['Date'].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)
    predicted_data = pd.DataFrame({'Date': [next_date], 'Close': [prediction]})

    combined_data = pd.concat([data, predicted_data])

    # Plot predicted data
    st.subheader('Closing Price with Prediction')
    fig2, ax2 = plt.subplots()
    ax2.plot(combined_data['Date'], combined_data['Close'])
    ax2.scatter(next_date, prediction, color='red', label='Predicted Price')
    ax2.legend()
    st.pyplot(fig2)

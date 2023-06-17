import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
from keras.models import load_model
import streamlit as st
import yfinance as yf


start_date = "2021-06-11"
end_date = "2023-06-11"

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker')
Info = yf.Ticker(user_input)
# Initializing variable for retrieving market prices
df = Info.history(start=start_date, end=end_date, period='1mo')
# Printing the historical market prices in the output
st.text("Market Prices data : ")
#df= df.drop('Dividends', 'Stock Splits', axis= 1)
#print(df)
#df= data.DataReader()
#df = data.DataReader(name="TSLA", data_source='yahoo', start=start_date, end=end_date)

#Describing Data
#st.subheader('Data from 2022 - 2023')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df, 'g')
st.pyplot(fig)


st.subheader('Closing Price vs Time chart WITH 100MA')
ma100 = df.rolling(8).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'b')
plt.plot(df, 'g')
st.pyplot(fig)


st.subheader('Closing Price vs Time chart WITH 100MA 200MA')
ma100 = df.rolling(8).mean()
ma200 = df.rolling(15).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'b')
plt.plot(ma200, 'r')
plt.plot(df, 'g')
st.pyplot(fig)

#Splitting Data into Training and Testing

df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

df_train_arr = scaler.fit_transform(df_train)
df_test_arr = scaler.fit_transform(df_test)


#Splitting Data into x_train and y_train
x_train = []
y_train = []
for i in range(100,df_train_arr.shape[0]):
    x_train.append(df_train_arr[i-100: i])
    y_train.append(df_train_arr[i,0])
#st.text(x_train)
x_train, y_train = np.array(x_train), np.array(y_train)


model = load_model('KerasModel.h5')

#Testing Part

past_100_days = df_train.tail(100)
final_df = past_100_days._append(df_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range (100,input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = model.predict(X_test)
scaler = scaler.scale_



scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_test  = y_test * scale_factor

#Final

st.subheader('Predictions for upcoming days')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)








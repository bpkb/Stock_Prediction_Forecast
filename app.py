# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:25:39 2023

@author: PKB
"""

# Import necessary libraries
from urllib.parse import urljoin

import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error
import math
import plotly.figure_factory as ff

# CSS style to hide Streamlit menu items and add a custom footer
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {
	
	visibility: hidden;
	
	}
footer:after {
	content:'Pranjal Bhatt'; 
	visibility: visible;
	display: block;
	position: relative;
# 	background-color: red;
	padding: 5px;
	top: 2px;
}
            </style>
            """
# Streamlit configuration settings
st.set_page_config(
    page_title="PKB_Stock_Prediction",
    # page_icon="🧊",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is an *extremely* cool app!"
    }
)

# Apply custom CSS styles
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Sidebar for selecting parameters
st.sidebar.markdown("## Select Parameters ")
country=['India', 'US', 'Canada']
stock = st.sidebar.selectbox("## Select Country", country)
snp500 = pd.DataFrame()

# Read stock data based on selected country
if(stock=='India'):
    snp500 = pd.read_csv("india_stock.csv",encoding='ISO-8859-1',header=None,skiprows=1)
elif(stock=='Canada'):
    # snp500.drop()
    snp500 = pd.read_csv("canada.csv",encoding='ISO-8859-1',header=None,skiprows=1)
    column_names = ['Name','Symbol']
    snp500.columns=column_names
elif(stock=='US'):
    # snp500.drop()
    snp500 = pd.read_csv("US_STOCKS.csv")
else:
    snp500 = pd.read_csv("US_STOCKS.csv")

# Set title and select stock
plt.style.use("fivethirtyeight")
st.title('Stock Market Forecasting')
column_names = ['Name', 'Symbol']
snp500.columns=column_names
stock = snp500['Name'].sort_values().tolist()

# Selecting stock from sidebar
stock = st.sidebar.selectbox("## Select Stock", stock)

# Function to find symbol of the selected stock
def find_symbol(stock):
    row = snp500[snp500['Name'] == stock]
    if not row.empty:
        return row['Symbol'].values[0]
    return "Symbol not found"

# Checking if stock is selected
if stock:
   msft = yf.Ticker(find_symbol(stock))
  # msft = yf.Ticker(stock)
else:
  msft=yf.Ticker('AAPL')

# Display company profile information in an expander
with st.expander("Company Profile"):
    try:
        if(msft.get_info()["sector"]):
            st.metric("Sector", msft.get_info()["sector"])
    except Exception:
        C=0
    try:
        if(msft.get_info()["industry"]):
            st.metric("Industry", msft.get_info()["industry"])
    except Exception:
        C=0
    try:
        if(msft.get_info()["website"]):
            st.metric("Website", msft.get_info()["website"])
    except Exception:
        C=0

# Fetch stock data based on selected date range
yf.pdr_override()


# Checking if stock is selected
if stock =='':
    stock='AAPL'

# Creating containers for selecting start and end date
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

today = datetime.date.today()
start_date = sub_columns[0].date_input('From', today-datetime.timedelta(days=1023))
end_date = sub_columns[1].date_input('To', today-datetime.timedelta(days=1))

# Validate selected dates
if start_date < end_date and end_date <= today:
    print("")
else:
    st.error('Error: End date must fall after start date or before current date.')

# Download stock data based on selected stock and dates
df = yf.download(find_symbol(stock), start=start_date, end=end_date)
# df = yf.download(stock, start=start_date, end=end_date)
df.index = df.index.date # Converting index to only contain dates

# Display stock data and performance history
if df.empty:
    st.error('Error: Enter Valid stock name.')
show_data = st.sidebar.checkbox('{} Data'.format(stock))

if show_data:
    st.title("{} Latest Data".format(stock))
    st.dataframe(df.tail(5))
    df.style.set_properties(**{'background-color': 'black',
                            'color': 'green'})

# Display graph of stock performance over time if selected
show_graph = st.sidebar.checkbox('History of {}\'s Performance Over Year'.format(stock))
group_labels = ['Close Price USD ($)']
if show_graph:
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='{}\'s Company "Close-Price" History Graph'.format(stock)))
    fig = px.line(df,x=df.index,y=df.columns,title="Performance of the stock")
    fig.layout.update(title_text='{}\'s Performance'.format(stock), xaxis_rangeslider_visible=True, xaxis_title='Year', yaxis_title='Price ($)')
    st.plotly_chart(fig)
 	
# Allow user to select number of forecast days
forecast_day=st.sidebar.selectbox('Select number of forecast Days', [10, 20, 30, 40, 50, 60])
# forecast_day = 10

# Preprocessing for LSTM model
df1=df['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
train_test_split=0.8
training_size=int(len(df1)*train_test_split)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# Define functions for creating dataset for LSTM model
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)

# Define LSTM model architecture
D1= 0.2
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(30,1)))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(D1))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])

# Train LSTM model
epochs= 10
batch_size=64
verbose=1
history = model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=batch_size,verbose=verbose)


# Predict stock prices using LSTM model
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transform back into the original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)



# Plot model performance
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))
look_back=30
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
dataset = pd.DataFrame()
dataset['data']=scaler.inverse_transform(df1).reshape(-1)
dataset['date']=df.index
dataset.set_index('date',inplace=True)
dataset['trainPredictPlot']=trainPredictPlot
dataset['testPredictPlot'] =testPredictPlot



# Perform forecasting
x_input=test_data[-30:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=30
i=0
now=datetime.datetime.now()
print(f"Time: {now}")
while(i<forecast_day):

    if(len(temp_input)>30):
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,31)
day_pred=np.arange(31,forecast_day+1)

# Prepare DataFrame for forecasted prices
#Forecasting part
import datetime
a1=[]
today = datetime.date.today()
for i in range(0,forecast_day):
    a1.append(today + datetime.timedelta(days = i))

d=scaler.inverse_transform(lst_output).reshape(-1)

list_of_tuples = list(zip(a1,d))

df5= pd.DataFrame(list_of_tuples, columns = ['Date','forecast_price'])
df5=df5.set_index('Date')

dataset['train_data']=trainPredictPlot.reshape(-1)
dataset['test_data'] =testPredictPlot.reshape(-1)
dataset['date']=df.index
dataset.set_index('date',inplace=True)


data1 = pd.DataFrame()
a=len(df5['forecast_price'])
b=[]
for i in range(0, (a)):
    b.append(str(i))
data1["Day"] = b
data1["Date"]= a1
data1["Price"] = df5['forecast_price'].values
data1["% Change"] = ((df5['forecast_price'].pct_change().values*100).round(2))
data2 = data1["% Change"].astype(str)
data1.drop('% Change', axis=1,inplace= True)
data1["% Change"]= data2[1:]+"%"
data1.set_index("Day",inplace=True)
data1 = data1.replace(np.nan, '', regex=True)





data3= data1


data1 = pd.DataFrame()
a=len(df5['forecast_price'])
b=[]
for i in range(1, (a)+1):
    # b.append('T'+str(i))
    b.append(str(i))
data1["Day"] = b
data1["Date"]= a1
data1["Predicted Price ($)"] = df5['forecast_price'].values
data1["% Change"] = ((df5['forecast_price'].pct_change().values*100).round(2))
data2 = data1["% Change"].astype(str)
data1.drop('% Change', axis=1,inplace= True)
data1["% Change"]= data2[1:]+"%"
data1.set_index("Day",inplace=True)
data1 = data1.replace(np.nan, '', regex=True)

# Plot forecasted prices
st.title("{}'s Forecast for next {} Days".format(stock,forecast_day))
st.dataframe(data1)
fig=px.line(data1,x = data1.Date,y=data1['Predicted Price ($)'],title=' {}\'s Company upcoming price movement prediction'.format(stock))
st.plotly_chart(fig)


# Plot model performance
dataset = dataset.rename(columns={'data': 'Actual Values', 'testPredictPlot': 'Predicted Values'})

fig = px.line(dataset, x=dataset.index, y=['Actual Values','Predicted Values'], title="Model Performance")
fig.update_traces(
    line=dict(color=['red', 'green']),  # Change colors as needed
    selector=dict(type='scatter')
)
fig.update_layout(
    yaxis_title='Close Price USD ($)',
    xaxis_title='Date',
    xaxis_rangeslider_visible=True,
    title_text='Model Performance'
)
st.plotly_chart(fig)

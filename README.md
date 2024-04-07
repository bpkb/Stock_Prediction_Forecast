# Stock_Prediction_Forecast
# -*- coding: utf-8 -*-
Created on Sun Aug 13 19:25:39 2023

@author: PKB17

Tools Used
  Streamlit
  
  Requests
  
  YFinance
  
  Pandas
  
  NumPy
  
  Matplotlib
  
  Plotly
  
  DateTime
  
  MinMaxScaler
  
  TensorFlow
  
  Scikit-learn

Description
This project focuses on stock market forecasting using LSTM neural networks. It allows users to select a country and a stock to analyze. The application provides insights into the selected company's sector, industry, and website. It also displays the latest stock data and a historical performance graph. Users can choose the number of forecast days to predict future stock prices.

Setup
Install the required libraries:

bash
Copy Code
pip install streamlit requests yfinance pandas numpy matplotlib plotly scikit-learn tensorflow
Run the Streamlit app:

bash
Copy Code
streamlit run your_file.py
Usage
Select the country and stock from the sidebar.

Explore the company profile details.

Choose the date range for data analysis.

View the latest stock data and historical performance graph.

Select the number of forecast days for predicting future stock prices.

Forecast Results
The application provides a forecast for the selected stock for the next specified number of days. It displays the predicted prices and percentage change in a tabular format. Additionally, it visualizes the predicted price movement over the forecast period.

For a detailed analysis, refer to the Streamlit application code provided above.


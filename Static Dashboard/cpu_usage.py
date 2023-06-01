import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import random
from pmdarima.arima import auto_arima
import matplotlib.patches as mpatches
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
from PIL import Image


def cpu_usage_dashboard():
    # Create tabs for CPU Usage page
    tab1, tab2 = st.tabs(["Rolling Window of CPU Usage", "Timeseries Model of CPU Usage"])

    with tab1:
        #st.title("Rolling Window of CPU Usage")
        
        #st.write("This is the content for CPU Usage - Tab 1.")
        data = pd.read_csv("Raw_Data_CPU.csv", skiprows=[0,1,2])
        # data

        # merging date & time in one column
        data['Datetime'] = data['DATE / TIME'].astype(str) + ' ' + data['Unnamed: 1'].astype(str)
        # data

        # dropping unnecessary columns
        data = data.drop(columns=["DATE / TIME", "Unnamed: 1","Max CPU Load","Average CPU Load"], axis=0)
        # data

        #changing column name
        data.rename(columns={'Min CPU Load': 'CPU_Usage'}, inplace=True)

        data_list = ['Datetime', 'CPU_Usage']
        data = data[data_list]
        # data

        # Proper rows
        data_list = ['Datetime', 'CPU_Usage']
        data = data[data_list]
        # data

        #Converted in datetime format
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d-%b-%y %I:%M %p')
        # data

        #Adding title & Subheading
        def main():
            #st.title('5Min/Raw CPU Data Analysis')
            st.subheader('CPU Usage Dataframe:')
        if __name__ == '__main__':
            main()

        # Filling null values using linear interpolation method 
        data['CPU_Usage'].interpolate(method='linear', inplace=True)
        data = data.dropna()
        data


        # Calculating rolling mean and standard deviation
        rolling_window = data.rolling(window=12)
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()

        # Creating a scatter plot with upper and lower bounds
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['CPU_Usage'], mode='markers', name='CPU usage'))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean['CPU_Usage'] + (2 * rolling_std['CPU_Usage']), 
                                mode='lines', line=dict(color='green'), name='Upper Bound'))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean['CPU_Usage'] - (2 * rolling_std['CPU_Usage']), 
                                mode='lines', line=dict(color='green'), name='Lower Bound'))

        # Adding title and axis labels
        fig.update_layout(title='5Min/Raw CPU Data Analysis', xaxis_title='Datetime', yaxis_title='CPU Usage')

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)




        
        # Calculate rolling mean and standard deviation
        rolling_mean = data['CPU_Usage'].rolling(window=12).mean()
        rolling_std = data['CPU_Usage'].rolling(window=12).std()

        # Create a column for anomalies
        data['Anomaly'] = np.where((data['CPU_Usage'] < rolling_mean - 2*rolling_std) | (data['CPU_Usage'] > rolling_mean + 2*rolling_std), -1, 1)

        # Create a scatter plot with different colors for anomalies and normal values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['CPU_Usage'], mode='markers', 
                                marker=dict(color=np.where(data['Anomaly']==-1, 'red', 'green')), name='CPU usage'))

        # Adding title and axis labels
        fig.update_layout(title='CPU Usage Anomaly Detection', xaxis_title='Datetime', yaxis_title='CPU Usage')

        # Displaying the plot in Streamlit
        st.plotly_chart(fig)


    with tab2:
        
        #st.title("Timeseries Model of CPU Usage")
        #st.write("This is the content for CPU Usage - Tab 2.")
        # Load data
        data = pd.read_excel("5_Minute_CPU_Usage_Data.xlsx", index_col='Datetime', parse_dates=True)
        st.subheader('CPU Usage Dataframe')
        st.dataframe(data)

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data, model='additive', period=24)

        # Fit an ARIMA model to the data
        model = ARIMA(data, order=(1, 1, 1))
        results = model.fit()

        # Forecast future values
        forecast = results.forecast(steps=24)

        # Create Plotly figures
        fig_orig = go.Figure()
        fig_resid = go.Figure()
        fig_trend = go.Figure()
        fig_seasonal = go.Figure()
        fig_forecast = go.Figure()

        # Add original data trace to original plot
        fig_orig.add_trace(go.Scatter(x=data.index, y=data['CPU_Usage'], name='Original'))

        # Add residual trace to residual plot
        fig_resid.add_trace(go.Scatter(x=data.index, y=decomposition.resid, name='Residuals'))

        # Add trend trace to trend plot
        fig_trend.add_trace(go.Scatter(x=data.index, y=decomposition.trend, name='Trend'))

        # Add seasonal trace to seasonal plot
        fig_seasonal.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, name='Seasonal'))

        # Add forecast trace to forecast plot
        fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))

        # Update layout for each plot
        fig_orig.update_layout(title='Original Data', xaxis_title='Date', yaxis_title='CPU Usage')
        fig_resid.update_layout(title='Residuals', xaxis_title='Date', yaxis_title='Residuals')
        fig_trend.update_layout(title='Trend', xaxis_title='Date', yaxis_title='Trend')
        fig_seasonal.update_layout(title='Seasonal', xaxis_title='Date', yaxis_title='Seasonal')
        fig_forecast.update_layout(title='Forecast', xaxis_title='Date', yaxis_title='CPU Usage')

        # Create the Streamlit app
        #st.title("CPU Usage Data")
        #st.write("Original Data:")
        st.plotly_chart(fig_orig)
        #st.write("Residuals:")
        st.plotly_chart(fig_resid)
        #st.write("Trend:")
        st.plotly_chart(fig_trend)
        #st.write("Seasonal:")
        st.plotly_chart(fig_seasonal)
        #st.write("Forecast:")
        st.plotly_chart(fig_forecast)
# Anomaly plot: It will have a user selection, user can select caption Id , starttime and endtime(X-Axis), column name(Y-Axis)
#Filter the dataframe(original) and anomaly dataframe based on user selection.
#show the filtered data in the dashboard
#plot graph for the filtered data, Original:line plot, anomaly: scatter plot(red)
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
#

def memory_usage_dashboard():
    # Create tabs for Memory Usage page
    tab1, tab2, tab3 = st.tabs(["EDA of Memory Usage", "Rolling Window of Memory Usage", "Timeseries Model of Memory Usage"])

    with tab1:
        #st.title("Memory Usage Data Analysis")
        
        #st.write("This is the content for Memory Usage - Tab 1.")
        # Data load 
        data = pd.read_excel("new_data_preprocessed.xlsx")

        # App title
        #st.markdown("<h1><u>Memory Usage Data Analysis</u></h1>", unsafe_allow_html=True)

        # Memory Usage Dataframe
        st.subheader('Memory Usage Dataframe')
        st.dataframe(data)

        st.header("Data visualization findings")

        # Graph 1 - Line chart with 10 unique IP addresses
        st.subheader('1. Line Chart with 10 unique IP Addresses')
        unique_ips = data['IP Address'].unique()[:10]
        plot_data = data[data['IP Address'].isin(unique_ips)]
        plot_data = plot_data.groupby(['IP Address', 'Timestamp'])['Minimum Memory Used'].mean().reset_index()
        plot = alt.Chart(plot_data).mark_line().encode(
        x='Timestamp:T',
        y='Minimum Memory Used:Q',
        color='IP Address:N'
        ).properties(
        width=800,
        height=500
        )
        st.altair_chart(plot, use_container_width=True)
        st.write("This graph shows the minimum memory used over time for the 10 unique IP addresses.")
        st.write("- The x-axis represents the timestamp of memory usage, the y-axis represents the minimum memory used, and each line represents a unique IP address.")
        st.write("- In this graph IP Address 10.16.11.22 is showing unpredictable or inconsistent patterns, rather than following a consistent or expected trend.")
        st.write("   - It has the Maximum Memory Used as compare to other IP Addresses i.e. 27.23")
        # Graph 2 - Line chart with 10 unique IP addresses for 24 hours data

        st.subheader('2. Line Chart with 10 unique IP Addresses for 24 Hours Data')
        unique_ips = data['IP Address'].unique()[:10]
        plot_data = data[data['IP Address'].isin(unique_ips)]
        plot_data = plot_data[plot_data['Timestamp'].dt.date == plot_data['Timestamp'].max().date()]
        plot = alt.Chart(plot_data).mark_line().encode(
            x='Timestamp:T',
            y='Minimum Memory Used:Q',
            color='IP Address:N'
        ).properties(
        width=800,
        height=500
        )
        st.altair_chart(plot, use_container_width=True)
        st.write("This graph shows the minimum memory used by the 10 unique IP addresses in the past 24 hours.")
        st.write("- The x-axis represents the timestamp of memory usage, the y-axis represents the minimum memory used, and each line represents a unique IP address.")
        st.write("- Here all the IP Addresses having similar or consistent pattern except for 10.16.11.22, it has arising pattern over the time.")
        #Graph 3: Stacked bar chart
        st.subheader('3. Stacked Bar Chart with IP Addresses and Minimum Memory Used by Weekday')
        plot_data = data.groupby(['IP Address', 'weekday'])['Minimum Memory Used'].mean().reset_index()
        plot_data['pct'] = plot_data.groupby('IP Address')['Minimum Memory Used'].apply(lambda x: x/x.sum())
        plot = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('IP Address:N', sort=alt.EncodingSortField(field='Minimum Memory Used', order='ascending')),
            y='pct:Q',
            color=alt.Color('weekday:N', scale=alt.Scale(scheme='category10'))
        ).properties(
            width=800,
            height=500
        )
        st.altair_chart(plot, use_container_width=True)
        st.write("Stacked Bar Chart with IP Addresses and Minimum Memory Used by Weekday")
        st.write("- For all the IP Address the weekdays are showing similar patterns except for 10.11.16.22 server, it is showing slightly different patterns")
        st.write("- Its largest Memory usage is coming on Wednesday and Tuesday, We can also see on thursday and Friday usage become little less compare to other servers.")

    with tab2:
        
        #st.title("Rolling Window of Memory Usage")
        #st.write("This is the content for Memory Usage - Tab 2.")
        # Load the data
        df = pd.read_csv('new_data_preprocessed.csv', parse_dates=True, index_col='Timestamp')

        st.subheader('Memory Usage Dataframe')
        st.dataframe(df)

        # Get the unique IP Addresses
        unique_ips = df['IP Address'].unique()

        # Add a selectbox widget to get the IP Address input from the user
        ip_address = st.selectbox('Select the IP Address:', unique_ips)

        # Filter the dataframe based on the selected IP Address
        df = df[df['IP Address'] == ip_address]

        # Show the original data
        st.write('**Original Data:**')
        st.write(df)

        st.write('---')

        window_size = st.slider('Select the rolling window size:', min_value=2, max_value=24, step=1, value=12)

        st.write('---')

        # Create a rolling window of size window_size
        rolling_window = df.rolling(window=window_size)

        # Calculate the rolling mean and standard deviation
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()

        # Reset the index of the DataFrame
        rolling_mean.reset_index(inplace=True)
        rolling_std.reset_index(inplace=True)

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Minimum Memory Used'], color='blue', label='Original Data')
        ax.plot(rolling_mean['Timestamp'], rolling_mean['Minimum Memory Used'], color='red', label='Rolling Mean')
        ax.plot(rolling_mean['Timestamp'], rolling_mean['Minimum Memory Used'] + (2 * rolling_std['Minimum Memory Used']), color='green', label='Upper Bound')
        ax.plot(rolling_mean['Timestamp'], rolling_mean['Minimum Memory Used'] - (2 * rolling_std['Minimum Memory Used']), color='green', label='Lower Bound')
        ax.legend(loc='best')

        # Set the x-axis labels to display dates in a readable format
        date_form = mdates.DateFormatter("%m-%d %H:%M:%S")
        ax.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45)

        # Set the title and axis labels
        ax.set_title('Minimum Memory Used')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Minimum Memory Used')

        # Display the rolling window plot in Streamlit
        st.subheader('Rolling Window Plot:')
        st.pyplot(fig)

        st.write('---')


    with tab3:
        
        #st.title("Timeseries Model of Memory Usage")
        #st.write("This is the content for Memory Usage - Tab 3.")
        # Load data
        df = pd.read_excel("new_data_preprocessed.xlsx", parse_dates=True, index_col='Timestamp')
        st.set_option('deprecation.showPyplotGlobalUse', False)


        # Remove outliers using z-score method
        mean = df["Minimum Memory Used"].mean()
        std = df["Minimum Memory Used"].std()
        upper_bound = mean + 2 * std
        lower_bound = mean - 2 * std
        anomalies = df[(df["Minimum Memory Used"] > upper_bound) | (df["Minimum Memory Used"] < lower_bound)]
        df['Minimum Memory Used'] = df['Minimum Memory Used'].replace(anomalies['Minimum Memory Used'].values, np.NaN)

        # Interpolate missing values
        df['Minimum Memory Used'] = df['Minimum Memory Used'].interpolate()

        # Filter data for a specific IP Address
        df = df[df['IP Address'] == '10.16.11.252']

        # Convert 'IP Address' column to numerical values
        df['IP Address'] = df['IP Address'].apply(lambda x: int(x.split('.')[-1]))

        df.drop(['IP Address','weekday'], axis=1, inplace=True)

        # Decompose time series into trend, seasonality, and residuals
        decomposition = seasonal_decompose(df, model='multiplicative', period=8)

        # Create a copy of the data frame for ARIMA modeling
        df1 = df.copy()
        df1['residuals'] = decomposition.resid
        df1 = df1.fillna(0)
        df1['Final_value'] = df1['Minimum Memory Used'] - df1['residuals']

        # Fit an ARIMA model to the data and make predictions
        X = auto_arima(df1['Final_value'], seasonal=True, suppress_warnings=True)
        arima_model = sm.tsa.ARIMA(df1['Final_value'], order=X.order)
        pred = arima_model.fit()

        # Create a Streamlit app to display the dashboard
        #st.title("Memory Usage Dashboard")

        # Add the data frame to the dashboard
        st.subheader("Data Frame")
        st.write(df)

        # Add the decomposition plots to the dashboard
        plt.figure(figsize=(15,8))
        plt.subplot(411)
        plt.plot(df, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(decomposition.seasonal,label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        st.subheader("Decomposition Plots")
        st.pyplot()






        # # # Add the ARIMA prediction plots to the dashboard
        # plt.figure(figsize=(20,8))
        # df1['Final_value'][1:].plot()
        # pred.predict()[1:].plot()
        # pred.forecast(24)
        # #user_input = st.slider("Select the number of hours to forecast", min_value=1, max_value=24)
        # forecast_values = []
        # for i in range(24):
        #     model = auto_arima(df1['Final_value'], seasonal=True, suppress_warnings=True)
        #     forecast3 = model.predict(n_periods=1)
        #     forecast_values.append(forecast3.iloc[0])
        #     next_date = df1.index[-1] + pd.Timedelta(hours=1)
        # df1 = df1.append(pd.DataFrame((forecast3.iloc[0]), index=[next_date], columns=['Final_value']))
        # plt.figure(figsize=(20,8))
        # df1['Final_value'][1:278].plot()
        # pred.predict()[1:].plot()
        # df1['Final_value'].iloc[279:].plot()
        # plt.plot(df1['Final_value'].iloc[-24:], label="Forecasted Values")
        # plt.legend(loc='best')
        # plt.title("ARIMA Model Forecast for Next 24 hours")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Memory Usage")
        # plt.tight_layout()
        # st.subheader("ARIMA Prediction Plots")
        # st.pyplot()
        st.header('ARIMA Model Plot')
        image = Image.open('C:/Users/SaSingh/Desktop/Model.png')
        st.image(image, caption='ARIMA plot', use_column_width=True)
        #add legend to codes generate new image nad insert the new image


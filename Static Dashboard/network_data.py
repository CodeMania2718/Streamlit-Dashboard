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

def network_data_dashboard():
    st.title("Network Traffic Data Analysis")
    dict1 = {"mbps": 1024, "kbps": 1, "bps": 0.001, "gbps": 125000}

    data= pd.read_excel('Report_95th_Percentile_Traffic_-_Last_Month.xlsx', skiprows=[0,1])
    data["Receive pbs"] = data["Receive pbs"].apply(lambda x: float(x.split()[0])*dict1[x.lower().split()[1]])
    data["Transmit bps"] = data["Transmit bps"].apply(lambda x: float(x.split()[0])*dict1[x.lower().split()[1]])
    data["Total bps"] = data["Total bps"].apply(lambda x: float(x.split()[0])*dict1[x.lower().split()[1]])

    def main():
        #st.title('Network Traffic data Analysis')
        st.subheader('#Dataframe:')
    if __name__ == '__main__':
        main()

    data


    def main():
        st.subheader('#Data Analysis:')
    if __name__ == '__main__':
        main()

        # st.subheader('#Data Analysis:')

        # Grouping data by Interface Type
        df1 = data.groupby(['Interface Type']).sum()

        # Dropdown widget to select y-axis column
        y_column = st.selectbox('Select a column for the y-axis', options=['Total bps', 'Transmit bps', 'Receive pbs'])
        
        # Bar plot by Interface Type
        fig = px.bar(df1, x=df1.index, y=y_column, title=f'Bar plot by Interface Type (for {y_column}):')
        fig.update_xaxes(title='Interface Type')
        fig.update_yaxes(title=y_column)
        st.plotly_chart(fig)

        # Line plot by Interface Type
        fig = px.line(df1, x=df1.index, y=y_column, title=f'Line plot by Interface Type (for {y_column}):')
        fig.update_xaxes(title='Interface Type')
        fig.update_yaxes(title=y_column)
        st.plotly_chart(fig)

        # Grouping data by Vendor
    df2 = data.groupby(['Vendor']).sum()

        # Dropdown widget to select y-axis column
    y_column = st.selectbox('Select a column for the y-axis', options=['Transmit bps', 'Total bps', 'Receive pbs'])
        
        # Bar plot by Vendor
    fig = px.bar(df2, x=df2.index, y=y_column, title=f'Bar plot by Vendor (for {y_column}):')
    fig.update_xaxes(title='Vendor')
    fig.update_yaxes(title=y_column)
    st.plotly_chart(fig)

        # Line plot by Vendor
    fig = px.line(df2, x=df2.index, y=y_column, title=f'Line plot by Vendor (for {y_column}):')
    fig.update_xaxes(title='Vendor')
    fig.update_yaxes(title=y_column)
    st.plotly_chart(fig)
# else:
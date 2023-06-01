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



def disk_usage_dashboard():
    # Create tabs for CPU Usage page
    st.title('Disk Usage Data Analysis')
    # Load data
    date_cols = ['Date/Time']
    feature_cols = ['Entity','Metric','Date/Time','Value']
    df = pd.read_csv("Disk - Raw data.csv",engine='python',skiprows=[0,1,2,3],index_col=False, 
                parse_dates=date_cols,
                skip_blank_lines=True,dayfirst=True,usecols=feature_cols)

    # Preprocess data
    df.columns = df.columns.str.lower().str.replace(' ','_').str.replace('/','')
    df = df.sort_values(by="datetime")
    df = df.reset_index(drop=True)
    df.rename(columns={'value':'actuals'},inplace=True)
    df.drop(['entity', 'metric'],axis=1,inplace=True)
    df.set_index('datetime',inplace=True)
    #st.header('Disk Usage Data Analysis')
    st.subheader('Disk Usage Dataframe')
    st.dataframe(df)

    st.header('Decomposition Plot')
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df, model='additive', period=8)

    # Create the decomposition plot
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))
    ax[0].plot(df, label='Original')
    ax[0].legend(loc='best')
    ax[1].plot(decomposition.trend, label='Trend')
    ax[1].legend(loc='best')
    ax[2].plot(decomposition.seasonal,label='Seasonality')
    ax[2].legend(loc='best')
    ax[3].plot(decomposition.resid, label='Residuals')
    ax[3].legend(loc='best')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)








    # from PIL import Image
    st.header('Actual Observations')
    image = Image.open('C:/Users/SaSingh/Desktop/actualvalues.png')
    st.image(image, caption='ARIMA plot', use_column_width=True)
    st.write('In this  one anamolous point has been added manually')

    st.header('Actuals V/S Predicted')
    image = Image.open('C:/Users/SaSingh/Desktop/predictedvalues.png')
    st.image(image, caption='Actual v/s Predicted', use_column_width=True)

    # st.header('Anomaly Plot')
    # image = Image.open('C:/Users/SaSingh/Desktop/predictedvalues.png')
    # st.image(image, caption='Anomaly', use_column_width=True)

    # import streamlit as st

    # Read the HTML file as a string
    with open("fig.html", "r") as f:
        html_string = f.read()

    # Display the HTML file in a Streamlit component
    st.header('Anaomaly Plot')
    st.components.v1.html(html_string, width=700, height=500)



import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import yaml
from sklearn.neighbors import LocalOutlierFactor
import system_profiler_worker1 as spw1
from urllib.parse import quote_plus
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px
import pandas as pd
import streamlit as st
from Detect_anomaly import *
from training_and_saving import *
import altair as alt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import seaborn as sns
import random
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
st.set_page_config(page_title="Memory Dashboard", page_icon=":guardsman:")



# Create the sidebar menu
menu_items = ['Automated Memory Dashboard', 'Historical Data Analysis']
selection = st.sidebar.selectbox('Memory Usage', menu_items)

# Define the content for each page
def page1():
    #st.write('This is page 1')
    # add content specific to page 1
    # Define function to preprocess data
    
        # Remove units from columns that have %, GB, MB, or KB values
    def preprocess_data(df):
        # Remove units from columns that have %, GB, MB, or KB values
        df = df.replace(to_replace=r'([KMGT]?B|%$)', value='', regex=True)
        # Convert all columns that have had their units removed to numeric datatype
        df = df.applymap(lambda x: pd.to_numeric(x, errors='ignore'))

        # Check if columns containing "time" and "memory" are present
        memory_cols = [col for col in df.columns if 'memory' in col.lower()]
        time_cols = [col for col in df.columns if 'time' in col.lower()]

        # Check if memory_cols and time_cols are not empty
        if not memory_cols:
            st.warning("Memory column not detected in the uploaded file. Please upload a memory file.")
            return None

        if not time_cols:
            st.warning("Time column not detected in the uploaded file.")
            return None

        # Check if there are multiple memory_cols and time_cols
        if len(memory_cols) > 1:
            memory_col = st.selectbox("Select the memory column", memory_cols, help="Select a column for anomaly detection.Don't leave the select box unselected")
        else:
            memory_col = memory_cols[0]

        if len(time_cols) > 1:
            time_col = st.selectbox("Select the time column", time_cols)
        else:
            time_col = time_cols[0]

        # # Check if 'col' column is present
        # col_cols = [col for col in df.columns if 'col' in col.lower()]
        # if col_cols:
        #     if len(col_cols) > 1:
        #         col = st.selectbox("Select the col column", col_cols)
        #     else:
        #         col = col_cols[0]
        #     df = df[[time_col, col, memory_col]]
        # else:
        #     col = None
        #     df = df[[time_col, memory_col]]

        # Check if any column name contains "caption", "IP", "VM", or "nodeid"
        cols = [col for col in df.columns if any(x in col.lower() for x in ["caption", "ip", "vm", "nodeid"])]
        if cols:
            if len(cols) > 1:
                col = st.selectbox("Select a column name", cols)
            else:
                col = cols[0]
            unique_values = sorted(df[col].unique().tolist())
            value = st.selectbox(f"Select a {col} value", [''] + unique_values, help=f"Choose a value to display data for a specific {col}")
            if value:
                df = df[df[col] == value]
            df = df[[time_col, col, memory_col]]
        else:
            col = None
            df = df[[time_col, memory_col]]

        # Convert time column to datetime datatype
        df[time_col] = pd.to_datetime(df[time_col])
        # Sort the DataFrame by the time column
        df = df.sort_values(by=time_col)

        return df





    # Define function to preprocess data
    def preprocessed(original_df):
        # Remove units from specified columns
        original_df = original_df.replace(to_replace=r'([KMGT]?B|%$)', value='', regex=True)

        # Convert specified columns to datetime datatype
        time_cols = [col for col in original_df.columns if 'time' in col.lower()]
        original_df[time_cols] = original_df[time_cols].apply(pd.to_datetime, errors='coerce')

        # Convert specified columns to numeric datatype
        memory_cols = [col for col in original_df.columns if 'memory' in col.lower()]
        cpu_cols = [col for col in original_df.columns if 'cpu' in col.lower()]
        disk_cols = [col for col in original_df.columns if 'disk' in col.lower()]
        original_df[memory_cols + cpu_cols + disk_cols] = original_df[memory_cols + cpu_cols + disk_cols].apply(pd.to_numeric, errors='coerce')

        # Fill None values with string 'None' to display in output
        original_df = original_df.fillna('None')

        return original_df

    # # Define function to perform KMeans clustering
    # def perform_kmeans(original_df, cluster_column, groupby_column, k=10, plot=True):
    #     gd = original_df.groupby(groupby_column).agg({cluster_column:['max', 'mean','var']})
    #     gd.columns=[f"{i}_{j}" for i,j in gd.columns]

    #     scaler = StandardScaler()
    #     X_norm = scaler.fit_transform(gd)

    #     kmeans = KMeans(random_state=42)
    #     visualizer = KElbowVisualizer(kmeans, k=k, metric='distortion', timings=1)
    #     visualizer.fit(X_norm)
    #     if plot:
    #         visualizer.show()

    #     kmeans = KMeans(n_clusters=visualizer.elbow_value_,random_state=42)
    #     kmeans.fit(X_norm)
    #     gd['KMeans']=kmeans.labels_
    #     return gd.reset_index(), kmeans
    # Define function to perform KMeans clustering
    def perform_kmeans(original_df, cluster_column, groupby_column, k=10, plot=True):
        #original_df = original_df.replace([np.inf, -np.inf], np.nan)
        gd = original_df.groupby(groupby_column).agg({cluster_column:['max', 'mean','var']})
        gd.columns=[f"{i}_{j}" for i,j in gd.columns]

        # Fill missing values with default value 0
        gd = gd.fillna(0)
    
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(gd)

        kmeans = KMeans(random_state=42)
        visualizer = KElbowVisualizer(kmeans, k=k, metric='distortion', timings=1)
        visualizer.fit(X_norm)
        if plot:
            visualizer.show()

        kmeans = KMeans(n_clusters=visualizer.elbow_value_,random_state=42)
        kmeans.fit(X_norm)
        gd['KMeans']=kmeans.labels_
        return gd.reset_index(), kmeans


    # Define function to select columns and display unique values
    def select_columns(original_df):
        st.write("Select the columns to display:")
        cols = st.multiselect("", original_df.columns.tolist())
        if cols:
            original_df = preprocessed(original_df[cols])
            unique_values = {}
            for col in cols:
                if 'IP' in col or 'caption' in col or 'nodeId' in col or 'VM' in col:
                    unique_values[col] = sorted(original_df[col].unique().tolist())
            if unique_values:
                for col, values in unique_values.items():
                    value = st.selectbox(f"Select a {col} value", [''] + values)
                    if value:
                        original_df = original_df[original_df[col] == value]
            return original_df


    def file_uploader():
        
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) # Replace with pd.read_excel for xlsx files
                original_df = df.copy() # Make a copy of the original dataframe
                df = preprocess_data(df) # Call preprocess_data function to preprocess data
                return original_df, df # Return original and processed dataframes
            except Exception as e:
                st.error(str(e))
                return None, None
        else:
            return None, None


    # Define main function
    def main():
        # Set page title and description
        #st.title("Memory File Uploader Dashboard")
        #st.markdown("<h1 style='text-decoration:underline;'>Fully Dynamic Memory Usage Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color:Black;font-weight:bold;font-size:35px'>Fully Dynamic Memory Usage Dashboard</span>", unsafe_allow_html=True)

        st.write("Select a Memory file to upload and view its data.")

        # Call file_uploader function to upload file and get processed DataFrame
        original_df, df = file_uploader()
        if df is not None:
            # Define variables inside main function to avoid error
            #memory_col = [col for col in df.columns if 'memory' in col.lower()][0]
            memory_cols = [col for col in df.columns if 'memory' in col.lower()]
            if not memory_cols:
                st.warning("Memory column not detected in the uploaded file. Please upload a memory file.")
                return None, None
            memory_col = memory_cols[0]

            
            time_col = [col for col in df.columns if 'time' in col.lower()][0]
            col = df.columns[1]

            #st.subheader("Original Dataframe")
            #st.write(original_df)

            st.subheader("Dataframe")
            st.write(df)
        


            # Add widgets for selecting time range and Y-axis column below the dataframe display
        
            st.markdown("<span style='color:Black;font-weight:bold;font-size:30px'>Anomaly Detection Settings</span>", unsafe_allow_html=True)

            # start_date = df[time_col].min().date()
            # end_date = df[time_col].max().date()
            # start_time = st.date_input("Start Date", start_date, min_value=start_date, max_value=end_date)
            # end_time = st.date_input("End Date", end_date, min_value=start_date, max_value=end_date)
            algorithm = st.selectbox("Select Anomaly Detection Algorithm", ["One-class SVM", "Rolling Window", "Isolation Forest", "Local Outlier Factor", "Auto ARIMA"])

            #condition to use which algorithm
            if algorithm == "One-class SVM":
                # # Filter anomaly data by time range
                # anomaly_filtered_df = Anomaly_Detection(df, memory_col)
                # #anomaly_filtered_df = df[(df[time_col[0]].dt.date >= start_time) & (df[time_col[0]].dt.date <= end_time)]
                # # anomaly_filtered_df = df[(df[time_col].dt.date >= start_time) & (df[time_col].dt.date <= end_time)]

                # # Display Output Dataframe
                # anomaly_filtered_df['Anomalities'] = np.where(anomaly_filtered_df['Anomaly'] == -1, 'Anomalies', 'Normal')
                # st.subheader(f"Anomaly Dataframe for {memory_col} using One Class SVM Algorithm")
                # #st.write(anomaly_filtered_df[[time_cols[0], filtered_df[filter_cols], y_col, 'Anomaly', 'Anomalities']])
                # # st.write(anomaly_filtered_df[[time_col, col, memory_col, 'Anomaly', 'Anomalities']])
                # st.write(anomaly_filtered_df)

                # st.subheader("Anomalies Detected Plot")
                # anomaly_points_df = anomaly_filtered_df[anomaly_filtered_df['Anomaly'] == -1]
                # combined_fig = go.Figure()
                # if not anomaly_points_df.empty:
                #     combined_fig.add_trace(go.Scatter(
                #         x=anomaly_points_df[time_col],
                #         y=anomaly_points_df[memory_col],
                #         mode='markers',
                #         marker=dict(
                #             color='red',
                #             size=5,
                #             line=dict(
                #                 width=0.2,
                #             )
                #         )
                #     ))
                # combined_fig.add_trace(go.Scatter(
                #     x=anomaly_filtered_df[time_col],
                #     y=anomaly_filtered_df[memory_col],
                #     mode='lines',
                #     line=dict(
                #         color='blue',
                #         width=1
                #     )
                # ))

                # st.plotly_chart(combined_fig)
                # Divide the data into train and test sets
                # sorted out with respect to timestamp
                
                train_size = int(0.8 * len(df))
                train_data = df[:train_size]
                test_data = df[train_size:]

                # Set the timestamp column as the index
                train_data.set_index(time_col, inplace=True)
                test_data.set_index(time_col, inplace=True)

                # Train the model
                model = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
                model.fit(train_data[[memory_col]])

                # Predict on the test data
                test_data['anomaly'] = pd.Series(model.predict(test_data[[memory_col]]), index=test_data.index)
                anomalies = test_data.loc[test_data['anomaly'] == -1]
                # Display the data frame
                st.write(test_data[[memory_col, 'anomaly']])

                # Plot the anomalies
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(test_data.index, test_data[memory_col], label='Memory usage')
                ax.scatter(anomalies.index, anomalies[memory_col], color='red', label='Anomaly')
                ax.legend(loc='upper left')
                plt.xticks(rotation=45)
                plt.title('One Class SVM Anomaly Detection Plot')
                plt.xlabel('Timestamp')
                plt.ylabel('Memory usage')
                st.pyplot(fig)

            elif algorithm == 'Rolling Window':

                window_size = st.slider("Window size", min_value=1, max_value=24)
                st.subheader(f"Anomalies Dataframe for {memory_col} using Rolling Window Method")

                # Create dataframe with selected column
                #selected_col_df = df[[time_col, memory_col, col]].copy()
                selected_col_df = df.copy()
                

                # Apply rolling window function
                selected_col_df['rolling_mean'] = selected_col_df[memory_col].rolling(window_size).mean()
                selected_col_df['rolling_std'] = selected_col_df[memory_col].rolling(window_size).std()

                # Calculate anomaly scores
                
                selected_col_df['anomaly_score'] = (selected_col_df[memory_col] - selected_col_df['rolling_mean']) / selected_col_df['rolling_std']
                # selected_col_df = selected_col_df[(selected_col_df[time_cols[0]].dt.date >= start_time) & (selected_col_df[time_cols[0]].dt.date <= end_time)]
                # selected_col_df['anomaly_score'] = selected_col_df['anomaly_score'].fillna(0)
                st.write(selected_col_df)

                # Create a CSV file path for export
                csv_file_path = "anomaly_scores.csv"

                # Export the dataframe to a CSV file
                selected_col_df[[time_col, memory_col, 'anomaly_score']].to_csv(csv_file_path, index=False)

                # Display a success message
                st.success(f"The data has been exported to {csv_file_path} successfully.")

                st.subheader("Plot")

                # Create scatter plot of anomaly scores
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=selected_col_df[time_col], y=selected_col_df[memory_col],
                                    mode='markers',
                                    marker=dict(color='green', size=5),
                                    name=f'{memory_col}'
                                ))
                # fig.add_trace(go.Scatter(x=selected_col_df[time_cols[0]], y=selected_col_df['rolling_mean'],
                #                     mode='lines',
                #                     name='Rolling Mean'
                #                 ))
                fig.add_trace(go.Scatter(x=selected_col_df[time_col], y=selected_col_df['rolling_mean'] + 2 * selected_col_df['rolling_std'],
                                    mode='lines',
                                    line=dict(color='blue'),
                                    name='Upper Bound'
                                ))
                fig.add_trace(go.Scatter(x=selected_col_df[time_col], y=selected_col_df['rolling_mean'] - 2 * selected_col_df['rolling_std'],
                                    mode='lines',
                                    line=dict(color='blue'),
                                    name='Lower Bound'
                                ))
                fig.add_trace(go.Scatter(x=selected_col_df[selected_col_df['anomaly_score'].abs() > 2][time_col], 
                                    y=selected_col_df[selected_col_df['anomaly_score'].abs() > 2][memory_col],
                                    mode='markers',
                                    marker=dict(color='red', size=5),
                                    name='Anomaly'
                                ))

                fig.update_layout(title=f"Anomaly Detection for {memory_col}")
                st.plotly_chart(fig)

                # Wrap the text description in an expander
                with st.expander("What is Anomaly Score?"):
                    st.write("Anomaly score is a measure of how many standard deviations away a point is from the mean. It is calculated by taking the difference between the value and the rolling mean, and dividing that by the rolling standard deviation. Points with high anomaly scores are likely to be anomalous.")
            elif algorithm == 'Isolation Forest':
                # Set the timestamp column as the index
                df = df.set_index(time_col)

                # Keep only the timestamp and memory columns
                df = df[[memory_col]]


                # Split the data into training and testing sets
                train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

                # Define the Isolation Forest model
                model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), random_state=42)

                # Fit the model on the training data
                model.fit(train_data)

                # Predict the anomalies in the test data
                predictions = model.predict(test_data)

                # Add the anomaly predictions as a new column in the test data
                test_data['anomaly'] = predictions

                st.write(test_data)

                # Separate the normal points and anomalies
                normal_points = test_data[test_data['anomaly'] == 1]
                anomaly_points = test_data[test_data['anomaly'] == -1]

                # Create a scatter plot of the anomalies
                fig, ax = plt.subplots()
                ax.scatter(normal_points.index, normal_points[memory_col], c='green', label='Normal')
                ax.scatter(anomaly_points.index, anomaly_points[memory_col], c='red', label='Anomaly')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('value')
                ax.set_title('Isolation Forest Anomaly Detection')
                ax.legend()
                plt.show()

                # Display the plot in the Streamlit app
                st.pyplot(fig)
            elif algorithm == 'Local Outlier Factor':
                # Set the timestamp as the index column
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col)

                # Divide the df into train and test sets (80% train, 20% test)
                train_size = int(len(df) * 0.8)
                train_data = df.iloc[:train_size]
                test_data = df.iloc[train_size:]

                # Select the columns to use (e.g., 'memory' column)
                train_features = train_data[memory_col].values.reshape(-1, 1)
                test_features = test_data[memory_col].values.reshape(-1, 1)

                # Apply the LOF algorithm
                lof = LocalOutlierFactor()
                train_scores = lof.fit_predict(train_features)
                test_scores = lof.fit_predict(test_features)
                # Add the anomaly predictions as a new column in the test data
                test_data['anomaly'] = test_scores

                # Display the DataFrame with the columns 'timestamp', 'memory', and 'anomaly'
                st.subheader("Data with Anomaly Predictions")
                st.write(test_data[[memory_col, 'anomaly']])

                # Plot the anomalies in the test data
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(test_data.index, test_data[memory_col], label='Memory Usage')
                ax.scatter(test_data.index[test_scores == -1], test_data[memory_col][test_scores == -1], color='red', label='Anomalies')
                ax.set_xlabel('Timestamp')
                ax.set_ylabel('Memory Usage')
                ax.set_title('Anomaly Detection using LOF')
                ax.legend()

                # Display the plot in the Streamlit app
                st.pyplot(fig)
            
            elif algorithm == 'Auto ARIMA':
                df.set_index(time_col, inplace=True)
                df.drop(df.columns[0], axis=1, inplace=True)
                # st.write(df)

                decomposition = seasonal_decompose(df, model='additive', period=8)
                df['residuals'] = decomposition.resid
                df=df.fillna(0)
                df['Final_value'] = df[memory_col] - df['residuals']
                df1= df.copy()
                # Drop multiple columns
                columns_to_drop = [memory_col, 'residuals']
                df1.drop(columns=columns_to_drop, inplace=True)
                from pmdarima.arima import auto_arima
                X = auto_arima(df1['Final_value'], seasonal=True, suppress_warnings=True)
                predictions = X.predict_in_sample(df1['Final_value'])
                predictions = pd.Series(predictions, index=df1['Final_value'].index)
                errors = df1['Final_value'] - predictions
                df1['errors'] = errors
                Q1 = df1['errors'].quantile(0.25)
                Q3 = df1['errors'].quantile(0.75)
                IQR = Q3 - Q1
                #

                df1['errors'] = df1['errors'][~((df1['errors'] < (Q1 - 1.5 * IQR)) | (df1['errors'] > (Q3 + 1.5 * IQR)))]
                window_size = st.slider("Window size", min_value=1, max_value=60)
                # st.subheader(f"Anomaly detection using Auto ARIMA")
                # Create a dataframe with selected columns
                selected_col_df = df1.copy()
                # Apply rolling window function
                selected_col_df['rolling_mean'] = selected_col_df['errors'].rolling(window_size).mean()
                selected_col_df['rolling_std'] = selected_col_df['errors'].rolling(window_size).std()


                # Calculate anomaly scores
                selected_col_df['anomaly_score'] = (selected_col_df['errors'] - selected_col_df['rolling_mean']) / selected_col_df['rolling_std']


                # Display the dataframe
                # selected_col_df[[time_col, errors', 'anomaly_score']]
                # Plot the data
                # fig = go.Figure()
                # fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['errors'],
                #                         mode='markers',
                #                         marker=dict(color='green', size=5),
                #                         name='errors'
                #                     ))
                # fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['rolling_mean'] + 2 * selected_col_df['rolling_std'],
                #                         mode='lines+markers',
                #                         line=dict(color='blue'),
                #                         name='Upper Bound'
                #                     ))
                # fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['rolling_mean'] - 2 * selected_col_df['rolling_std'],
                #                         mode='lines+markers',
                #                         line=dict(color='blue'),
                #                         name='Lower Bound'
                #                     ))
                # fig.add_trace(go.Scatter(x=selected_col_df[selected_col_df['anomaly_score'].abs() > 2].index, 
                #                         y=selected_col_df[selected_col_df['anomaly_score'].abs() > 2]['errors'],
                #                         mode='markers',
                #                         marker=dict(color='red', size=5),
                #                         name='Anomaly'
                #                     ))


                # fig.update_layout(title=f"Anomaly Detection using Auto ARIMA")
                # # fig.show()
                # st.plotly_chart(fig)
                import numpy as np

                # Interpolate missing values
                selected_col_df['rolling_mean_interp'] = selected_col_df['rolling_mean'].interpolate(method='linear')
                selected_col_df['rolling_std_interp'] = selected_col_df['rolling_std'].interpolate(method='linear')

                # Plot the data with interpolated values
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['errors'],
                                        mode='markers',
                                        marker=dict(color='green', size=5),
                                        name='errors'
                                    ))
                fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['rolling_mean_interp'] + 2 * selected_col_df['rolling_std_interp'],
                                        mode='lines+markers',
                                        line=dict(color='blue'),
                                        name='Upper Bound'
                                    ))
                fig.add_trace(go.Scatter(x=selected_col_df.index, y=selected_col_df['rolling_mean_interp'] - 2 * selected_col_df['rolling_std_interp'],
                                        mode='lines+markers',
                                        line=dict(color='blue'),
                                        name='Lower Bound'
                                    ))
                fig.add_trace(go.Scatter(x=selected_col_df[selected_col_df['anomaly_score'].abs() > 2].index, 
                                        y=selected_col_df[selected_col_df['anomaly_score'].abs() > 2]['errors'],
                                        mode='markers',
                                        marker=dict(color='red', size=5),
                                        name='Anomaly'
                                    ))

                fig.update_layout(title=f"Anomaly Detection using Auto ARIMA")
                st.plotly_chart(fig)

            st.markdown("<span style='color:Black;font-weight:bold;font-size:30px'>Exploratory Data Analysis</span>", unsafe_allow_html=True)
            #Display original dataframe
            st.subheader("Original Dataframe")
            with st.expander("ℹ️ Information"):
                st.write("This is the uploaded data for EDA to explore data insights.")
            st.write(original_df)
            original_df = preprocessed(original_df) 
            
            # Allow user to select column to cluster and group by
            cluster_column = st.selectbox("Select column to cluster", options=original_df.columns)
            groupby_column = st.selectbox("Select column to group by", options=original_df.columns)

            # Convert selected column to numeric datatype
            original_df[cluster_column] = pd.to_numeric(original_df[cluster_column], errors='coerce')

            # Perform KMeans clustering and display results
            st.subheader(f"KMeans Clustering on {cluster_column}")
            clustered_df, kmeans_model = perform_kmeans(original_df, cluster_column, groupby_column)
            st.write(clustered_df)
            original_df = select_columns(original_df)
            st.write(original_df)
            # Display summary statistics
            st.subheader("Summary Statistics")
            st.write(original_df.describe())
            # Filter columns by name
            time_cols = [col for col in original_df.columns if 'time' in col.lower()]
            memory_cols = [col for col in original_df.columns if 'memory' in col.lower()]
            cpu_cols = [col for col in original_df.columns if 'cpu' in col.lower()]
            disk_cols = [col for col in original_df.columns if 'disk' in col.lower()]
            #Id_cols = [col for col in df.columns if 'IP' in col or 'caption' in col or 'nodeId' in col or 'VM' in col.lower()]

            # Create new dataframe with selected columns
            selected_cols = time_cols + memory_cols + cpu_cols + disk_cols 
            filtered_df = original_df[selected_cols]

            # Add widgets for selecting time range and Y-axis column
            st.subheader("Graph Settings")
            start_date = filtered_df[time_cols[0]].min().date()
            end_date = filtered_df[time_cols[0]].max().date()
            start_time = st.date_input("Start Date", start_date, min_value=start_date, max_value=end_date, help="Select start and end dates to plot a graph for any time range, and modify the dates to view the graph for different time ranges.")
            end_time = st.date_input("End Date", end_date, min_value=start_date, max_value=end_date)
            y_col = st.selectbox("Select a Column", [col for col in filtered_df.columns if col != time_cols[0]], help="Choose a column for graph analysis.")
            # Convert timestamp column to datetime format
            filtered_df[time_cols[0]] = pd.to_datetime(filtered_df[time_cols[0]])

            # Filter selected data by time range
            filtered_df = filtered_df[(filtered_df[time_cols[0]].dt.date >= start_time) & (filtered_df[time_cols[0]].dt.date <= end_time)]
            #st.write(filtered_df)
            # Create line chart for Original Data plot
            fig = px.line(filtered_df, x=time_cols[0], y=y_col)

            # Display Original Data plot
            st.subheader(f"{y_col} vs. Time")
            st.plotly_chart(fig)



    # Call main function to run the app
    if __name__ == "__main__":
        main()

def page2():
    #st.write('This is page 2')
    import plotly.express as px

    # Load data from SQL Server into a pandas dataframe
    @st.cache_data
    def load_data():
        import joblib
        df = joblib.load('sql_data.jb')
        df=spw1.preprocess_df(df)
        rename_dict={'percentmemoryused':'average_percent_memory_used','nodeid':'caption'}
        df=spw1.rename_df(df,rename_dict)
        return df


    df = load_data()

    # Define function to prepare data for clustering
    def prepare_data(df, no_of_nodes):
        df1 = spw1.get_required_nodes(no_of_nodes, df)
        #df1 = spw1.filter_data_for_a_date(df1, date)
        gd1 = spw1.prepare_data_for_clustering(df1)
        X_norm1, scaler1 = spw1.standardize_data(gd1, ["average_percent_memory_used_max", "average_percent_memory_used_mean"])
        k1 = spw1.find_cluster_numbers(X_norm1)
        gd1, kmean_model1 = spw1.make_clusters(X_norm1, gd1, k1)
        # # Predict cluster labels for all data points using KMeans object
        # gd1['KMeans'] = kmean_model1.predict(X_norm1)
        return gd1
        # return gd1

    # Create Streamlit app
    # st.title("System Profiler Dashboard")
    #st.markdown("<h1><u>VM Profiling</u></h1>", unsafe_allow_html=True)
    st.markdown("<span style='color:Black;font-weight:bold;font-size:40px;text-decoration:underline'>VM Profiling</span>", unsafe_allow_html=True)

    # Allow user to select number of nodes
    st.subheader("Select number of nodes")
    no_of_nodes = st.slider(" ", min_value=100, max_value=6418, step=100, value=500)

    # Prepare data for clustering based on user selection
    gd1 = prepare_data(df, no_of_nodes)
    #gd1 = prepare_data(df, no_of_nodes, '2023-02-17')


    # Plot clusters using Plotly
    fig = px.scatter(gd1, x='average_percent_memory_used_mean', y='average_percent_memory_used_max', color='KMeans', color_continuous_scale=px.colors.sequential.Viridis)

    st.plotly_chart(fig)

    # Allow user to select a cluster
    st.subheader("**Select a Cluster**")
    selected_cluster = st.selectbox(" ", sorted(gd1['KMeans'].unique()))
    st.subheader("Dataframe & Cluster Stats")

    # Print dataframe for selected cluster
    cluster_df = gd1[gd1['KMeans']==selected_cluster][['caption', 'average_percent_memory_used_max', 'average_percent_memory_used_mean']]
    #st.write(cluster_df)


    # # Plot scatter plot for selected cluster
    # fig_cluster = px.scatter(cluster_df, x='average_percent_memory_used_max', y='average_percent_memory_used_mean', color='caption', color_continuous_scale=px.colors.sequential.Viridis)
    # st.plotly_chart(fig_cluster)

    # # Create dataframe with group and caption information
    cl1=pd.DataFrame(gd1.groupby('KMeans')['caption'].unique()).reset_index()[['KMeans','caption']]

    # Create dictionary with group and caption information
    gup=dict()
    for g,g_df in cl1.groupby('KMeans')["caption"]:
        gup[int(g)]=g_df[g]


        
    gsys=df[df['caption'].isin(gup[selected_cluster])].groupby("caption").agg({"average_percent_memory_used":['max', 'mean']})
    gsys.columns=[f"{i}_{j}" for i,j in gsys.columns]
    gsys,gsys.describe().T[['min','max','count']]


    # Plot scatter plot for selected cluster
    fig_cluster = px.scatter(cluster_df, x='average_percent_memory_used_mean', y='average_percent_memory_used_max', color='caption', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_cluster)




    # Get the list of captions available in all the four clusters
    common_captions = set(gd1[gd1['KMeans'].isin(range(4))]['caption'])
    #st.markdown("<span style='color:Black;font-weight:bold;font-size:30px'>Exploratory Data Analysis</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:Black;font-weight:bold;font-size:30px;text-decoration:underline'>Exploratory Data Analysis</span>", unsafe_allow_html=True)

    # Allow user to select a caption within the selected cluster
    st.subheader("**Select a Caption**")
    selected_caption = st.selectbox(" ", sorted(common_captions))

    # Filter data for selected caption
    caption_df = gd1[gd1['caption'] == selected_caption]

    # Get unique group numbers for selected caption
    group_numbers = caption_df['KMeans'].unique()

    # Print group numbers for selected caption
    if len(group_numbers) > 0:
        st.write(f"**The selected caption '{selected_caption}' belongs to group(s): {', '.join(str(num) for num in group_numbers)}**")
    else:
        st.write(f"No group found for the selected caption '{selected_caption}'")


    # Get the original data for selected caption
    original_data = df[df['caption'] == selected_caption]


    # Print original data for selected caption

    st.subheader(f"The original data for the selected caption '{selected_caption}' is:")
    st.write(original_data)

    # Select start and end datetime
    st.subheader("**Select Time Range**")
    start_datetime = pd.Timestamp(st.date_input("Start Date", original_data["timestamp"].min()))
    end_datetime = pd.Timestamp(st.date_input("End Date", original_data["timestamp"].max()))

    # Select column for Y-axis
    st.subheader("**Select Column for Y-axis**")
    y_column = st.selectbox("Column", list(original_data.columns))

    # Filter data based on selected time range
    time_filtered_data = original_data[(original_data["timestamp"] >= start_datetime) & (original_data["timestamp"] <= end_datetime)]

    # Plot data
    if y_column:
        st.subheader("**Plot**")
        fig = px.line(time_filtered_data, x="timestamp", y=y_column)
        st.plotly_chart(fig)
    else:
        st.write("Please select a column for Y-axis.")



    # # Load data for each group into a dictionary
    # data_dict = {}
    # for i in range(4):
    #     data_file = f"Memory_Data_Group{i}.csv"
    #     df = pd.read_csv(data_file)
    #     df2 = df.drop_duplicates(['caption', 'timestamp'])
    #     df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    #     df2.set_index('timestamp', inplace=True)
    #     df2.sort_index()
    #     threshold_date = pd.to_datetime('2/28/2023 23:00:00')
    #     train_data = df2.loc[df2.index <= threshold_date]
    #     test_data = df2.loc[df2.index > threshold_date]
    #     data_dict[i] = test_data

    # # Load models for each group into a dictionary
    # model_dict = {}
    # for i in range(4):
    #     model_file = f"modelgroup{i}SVM.pickle"
    #     with open(model_file, 'rb') as file:
    #         model = pickle.load(file)
    #     model_dict[i] = model

    # # Create a streamlit app
    # #st.title("Anomaly Detection")
    # st.markdown("<span style='color:Black;font-weight:bold;font-size:50px'>Anomaly Detection</span>", unsafe_allow_html=True)

    # # Allow user to select the group and caption to display
    # selected_group = st.selectbox("Select Group", [0, 1, 2, 3])
    # selected_data = data_dict[selected_group]
    # selected_caption = st.selectbox("Select Caption", selected_data['caption'].unique())

    # # Allow user to select date range for display
    # start_date = st.date_input("Select Start Date", selected_data.index.min())
    # end_date = st.date_input("Select End Date", selected_data.index.max())

    # # Convert start_date and end_date to datetime objects
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)

    # # Filter data based on user selections
    # mask = (selected_data['caption'] == selected_caption) & (selected_data.index >= start_date) & (selected_data.index <= end_date)
    # filtered_data = selected_data.loc[mask]

    
    # # Use the selected model to predict anomalies and plot the data
    # model = model_dict[selected_group]
    # features = ['average_percent_memory_used']
    # predictions = model.predict(filtered_data[features])
    # filtered_data['Anomaly'] = predictions
    # anomalies = filtered_data[filtered_data['Anomaly'] == -1]



    # import plotly.express as px

    # fig = px.line(filtered_data, x=filtered_data.index, y='average_percent_memory_used', title=f"{selected_caption} - {start_date} to {end_date}")
    # fig.add_scatter(x=anomalies.index, y=anomalies['average_percent_memory_used'], mode='markers', marker=dict(color='red'), name='Anomaly')
    # fig.update_layout(xaxis_title='Timestamp', yaxis_title='Average Percent Memory Used', legend_title='Legend')
    # st.plotly_chart(fig)




# Display the appropriate page based on user selection
if selection == 'Automated Memory Dashboard':
    page1()

else:
    page2()




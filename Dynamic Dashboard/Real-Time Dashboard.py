# import ast
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# #st.set_page_config(page_title="Forecast Dashboard", page_icon=":guardsman:", layout="wide")

# # Define the list of available captions
# CAPTION_LIST = [54, 1094, 15355, 16603, 16604, 16605, 16606, 16607, 16608, 16609, 16610, 16613, 16614, 16615, 16618, 16619, 16620, 16677, 20644, 20668]

# # Create a multi-select list of the available captions
# selected_captions = st.multiselect("Select Captions", CAPTION_LIST)

# if not selected_captions:
#     st.warning("Please select at least one caption.")
# else:
#     #Json file loading
#     dfs = []
#     for caption in selected_captions:
#         file_path = f'memory_data\memory_data\model_output_{caption}.csv'
#         # Load the JSON file into a Pandas DataFrame
#         forecast_df = pd.read_csv(file_path)
#         # Convert the forecast data from string to list
#         forecast_df['Forecast point data'] = forecast_df['Forecast point data'].apply(lambda x: ast.literal_eval(x))
#         dfs.append(forecast_df)

#     # Merge the selected DataFrames into a single DataFrame
#     forecast_df = pd.concat(dfs)

#     # Create empty lists for historical data and forecasts
#     historical_dfs = []
#     forecasts = []

#     # Create a plot using matplotlib
#     fig, axes = plt.subplots(len(selected_captions), 1, figsize=(10, 5 * len(selected_captions)))
#     fig.tight_layout(pad=5.0)
#     fig.subplots_adjust(hspace=0.5)

#     # Plot each selected caption's data in a separate subplot
#     for i, caption in enumerate(selected_captions):
#         ax = axes[i]
#         # Filter the DataFrame to get data for the current caption
#         filtered_df = forecast_df[forecast_df['caption'] == caption]
#         historical_df = [filtered_df['actual_Data'].iloc[0]]
#         historical_dfs.append(historical_df)
#         forecasts.append([filtered_df['Forecast point data'].iloc[0][0]])

#         ax.plot(historical_df, marker="o", color="red")
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Data')
#         ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')

#     # drop the first column
#     forecast_df = forecast_df.drop(forecast_df.columns[0], axis=1)

#     st.markdown("<span style='color:black;font-weight:bold;font-size:30px'>Plot:</span>", unsafe_allow_html=True)

#     # Display the plot and update it with each iteration
#     plot = st.pyplot(fig)
#     for i in forecast_df.index:
#         for j, caption in enumerate(selected_captions):
#             filtered_df = forecast_df[forecast_df['caption'] == caption]
#             ax = axes[j]
#             ax.clear()
#             ax.plot(historical_dfs[j], marker="o", color="red")
#             ax.plot(list(range(i+1, i+12+1)), filtered_df['Forecast point data'].iloc[i])
#             ax.plot(forecasts[j], marker='o', color="yellow")
#             historical_dfs[j].append(filtered_df['actual_Data'].iloc[i])
#             forecasts[j].append(filtered_df['Forecast point data'].iloc[i][0])
#             ax.set_xlabel('Time')
#             ax.set_ylabel('Data')
#             ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')
#         plot.pyplot(fig)
#         plt.pause(0.1)

#     plt.show()



# # # Create a multi-select list of the available captions
# # selected_captions = st.multiselect("Select Captions", CAPTION_LIST)

# # if not selected_captions:
# #     st.warning("Please select at least one caption.")
# # else:
# #     #Json file loading
# #     dfs = []
# #     for caption in selected_captions:
# #         file_path = f'memory_data\memory_data\model_output_{caption}.csv'
# #         # Load the JSON file into a Pandas DataFrame
# #         forecast_df = pd.read_csv(file_path)
# #         # Convert the forecast data from string to list
# #         forecast_df['Forecast point data'] = forecast_df['Forecast point data'].apply(lambda x: ast.literal_eval(x))
# #         dfs.append(forecast_df)

# #     # Merge the selected DataFrames into a single DataFrame
# #     forecast_df = pd.concat(dfs)

# #     # Create empty lists for historical data and forecasts
# #     historical_dfs = []
# #     forecasts = []

# #     # Create a plot using matplotlib
# #     fig, axes = plt.subplots(2, 2, figsize=(30, 20))
# #     fig.subplots_adjust(hspace=0.5, wspace=0.2)

# #     # Plot each selected caption's data in a separate subplot
# #     for i, caption in enumerate(selected_captions):
# #         row = i // 2
# #         col = i % 2
# #         ax = axes[row, col]
# #         # Filter the DataFrame to get data for the current caption
# #         filtered_df = forecast_df[forecast_df['caption'] == caption]
# #         historical_df = [filtered_df['actual_Data'].iloc[0]]
# #         historical_dfs.append(historical_df)
# #         forecasts.append([filtered_df['Forecast point data'].iloc[0][0]])

# #         # ax.plot(historical_df, marker="o", color="red")
# #         # ax.set_xlabel('**Time**')
# #         # ax.set_ylabel('**Average percent memory used**')
# #         # # ax.set_title(f'**Forecast vs. Actual Data for Caption {caption}'**)
# #         ax.set_xlabel('Time', fontsize=16, fontweight='bold')
# #         ax.set_ylabel('Average percent memory used', fontsize=16, fontweight='bold')
# #         ax.set_title(f'Forecast vs. Actual Data for Caption {caption}', fontsize=16, fontweight='bold')

# #     # drop the first column
# #     forecast_df = forecast_df.drop(forecast_df.columns[0], axis=1)

# #     st.markdown("<span style='color:black;font-weight:bold;font-size:30px'>Plot:</span>", unsafe_allow_html=True)

# #     # Display the plot and update it with each iteration
# #     plot = st.pyplot(fig)
# #     for i in forecast_df.index:
# #         for j, caption in enumerate(selected_captions):
# #             row = j // 2
# #             col = j % 2
# #             filtered_df = forecast_df[forecast_df['caption'] == caption]
# #             ax = axes[row, col]
# #             ax.clear()
# #             ax.plot(historical_dfs[j], marker="o", color="red")
# #             ax.plot(list(range(i+1, i+12+1)), filtered_df['Forecast point data'].iloc[i])
# #             ax.plot(forecasts[j], marker='o', color="yellow")
# #             historical_dfs[j].append(filtered_df['actual_Data'].iloc[i])
# #             forecasts[j].append(filtered_df['Forecast point data'].iloc[i][0])
# #             # ax.set_xlabel('**Time**')
# #             # ax.set_ylabel('**Average percent memory used**')
# #             # # ax.set_title(f'**Forecast vs. Actual Data for Caption {caption}**')
            # ax.set_xlabel('Time', fontsize=16, fontweight='bold')
            # ax.set_ylabel('Average percent memory used', fontsize=16, fontweight='bold')
            # ax.set_title(f'Forecast vs. Actual Data for Caption {caption}', fontsize=16, fontweight='bold')
# #         plot.pyplot(fig)
# #         plt.pause(0.0001)

# #     plt.show()
import ast
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Forecast Dashboard", page_icon=":guardsman:", layout="wide")
st.markdown("<span style='color:black;font-weight:bold;font-size:50px'>Forecast Data Summary</span>", unsafe_allow_html=True)
# Define the list of available captions
CAPTION_LIST = [1094, 15355, 16603, 16604, 16605, 16606, 16607, 16608, 16610, 16613, 16614, 16615, 16618, 16619, 16620, 16677, 20644, 20668]

# Create a multi-select list of the available captions
selected_captions = st.multiselect("**Select Captions**", CAPTION_LIST)

if not selected_captions:
    st.warning("Please select at least one caption.")
else:
    #Json file loading
    dfs = []
    max_data_points = float('inf')
    for caption in selected_captions:
        file_path = f'memory_data\memory_data\model_output_{caption}.csv'
        # Load the JSON file into a Pandas DataFrame
        forecast_df = pd.read_csv(file_path)
        # Convert the forecast data from string to list
        forecast_df['Forecast point data'] = forecast_df['Forecast point data'].apply(lambda x: ast.literal_eval(x))
        dfs.append(forecast_df)
        # Find the minimum number of available data points for the selected captions
        if len(forecast_df) < max_data_points:
            max_data_points = len(forecast_df)

    # Merge the selected DataFrames into a single DataFrame
    forecast_df = pd.concat(dfs)

    # Create empty lists for historical data and forecasts
    historical_dfs = []
    forecasts = []

    # # Create a plot using matplotlib
    # fig, axes = plt.subplots(len(selected_captions), 1, figsize=(10, 5 * len(selected_captions)))
    # fig.tight_layout(pad=5.0)
    # fig.subplots_adjust(hspace=0.5)

    # # Plot each selected caption's data in a separate subplot
    # for i, caption in enumerate(selected_captions):
    #     ax = axes[i]
    #     # Filter the DataFrame to get data for the current caption
    #     filtered_df = forecast_df[forecast_df['caption'] == caption]
    #     historical_df = [filtered_df['actual_Data'].iloc[0]]
    #     historical_dfs.append(historical_df)
    #     forecasts.append([filtered_df['Forecast point data'].iloc[0][0]])

    #     ax.plot(historical_df, marker="o", color="red")
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Data')
    #     ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')

    # # drop the first column
    # forecast_df = forecast_df.drop(forecast_df.columns[0], axis=1)

    # st.markdown("<span style='color:black;font-weight:bold;font-size:30px'>Plot:</span>", unsafe_allow_html=True)

    # # Display the plot and update it with each iteration
    # plot = st.pyplot(fig)
    # for i in range(max_data_points):
    #     for j, caption in enumerate(selected_captions):
    #         filtered_df = forecast_df[forecast_df['caption'] == caption]
    #         ax = axes[j]
    #         ax.clear()
    #         ax.plot(historical_dfs[j], marker="o", color="red")
    #         ax.plot(list(range(i+1, i+12+1)), filtered_df['Forecast point data'].iloc[i])
    #         ax.plot(forecasts[j], marker='o', color="yellow")

    #         historical_dfs[j].append(filtered_df['actual_Data'].iloc[i])
    #         forecasts[j].append(filtered_df['Forecast point data'].iloc[i][0])
    #         ax.set_xlabel('Time')
    #         ax.set_ylabel('Data')
    #         ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')
    #     plot.pyplot(fig)
    #     plt.pause(0.1)
    # plt.show()
    # Create a plot using matplotlib
    num_rows = len(selected_captions) // 2 + len(selected_captions) % 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
    fig.tight_layout(pad=5.0)
    fig.subplots_adjust(hspace=0.5)

    # Plot each selected caption's data in a separate subplot
    for i, caption in enumerate(selected_captions):
        row_idx = i // 2
        col_idx = i % 2
        ax = axes[row_idx, col_idx]

        # Filter the DataFrame to get data for the current caption
        filtered_df = forecast_df[forecast_df['caption'] == caption]
        historical_df = [filtered_df['actual_Data'].iloc[0]]
        historical_dfs.append(historical_df)
        forecasts.append([filtered_df['Forecast point data'].iloc[0][0]])

        ax.plot(historical_df, marker="o", color="red", label="Actual point")
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Data')
        # ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average percent memory used', fontsize=12, fontweight='bold')
        ax.set_title(f'Forecast vs. Actual Data for Caption {caption}', fontsize=12, fontweight='bold')
        # Add legend to the plot
        ax.legend()

    # drop the first column
    forecast_df = forecast_df.drop(forecast_df.columns[0], axis=1)

    st.markdown("<span style='color:black;font-weight:bold;font-size:30px'>Plot:</span>", unsafe_allow_html=True)

    # Display the plot and update it with each iteration
    plot = st.pyplot(fig)
    for i in range(max_data_points):
        for j, caption in enumerate(selected_captions):
            filtered_df = forecast_df[forecast_df['caption'] == caption]
            row_idx = j // 2
            col_idx = j % 2
            ax = axes[row_idx, col_idx]

            ax.clear()
            ax.plot(historical_dfs[j], marker="o", color="red", label="Actual point")
            ax.plot(list(range(i+1, i+12+1)), filtered_df['Forecast point data'].iloc[i])
            ax.plot(forecasts[j], 'm--', marker='o', color="yellow", label="Forecast point")

            historical_dfs[j].append(filtered_df['actual_Data'].iloc[i])
            forecasts[j].append(filtered_df['Forecast point data'].iloc[i][0])
            # ax.set_xlabel('Time')
            # ax.set_ylabel('Data')
            # ax.set_title(f'Forecast vs. Actual Data for Caption {caption}')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average percent memory used', fontsize=12, fontweight='bold')
            ax.set_title(f'Forecast vs. Actual Data for Caption {caption}', fontsize=12, fontweight='bold')
            # Add legend to the plot
            ax.legend()

        plot.pyplot(fig)
        plt.pause(0.1)
    plt.show()

import streamlit as st
#from Solarwinds_data import profiling_dashboard
from memory_usage import memory_usage_dashboard
from cpu_usage import cpu_usage_dashboard
from disk_usage import disk_usage_dashboard
from network_data import network_data_dashboard

import base64
st.set_page_config(page_title="Memory Dashboard", page_icon=":guardsman:")


def auth_app():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Authenticate the user here, for example by checking the username and password against a database
        if username == "admin" and password == "1234":
            # Store the authentication state in session state
            st.session_state.is_authenticated = True
            # Redirect to the dashboard app
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def dashboard_app():
    # selected_option = st.sidebar.selectbox("Select an option", SIDEBAR_OPTIONS)
    # Define sidebar options
    SIDEBAR_OPTIONS = ["Home", "Memory Usage", "CPU Usage", "Disk Usage", "Network Data"]
    selected_option = st.sidebar.selectbox("Select an option", SIDEBAR_OPTIONS)
    # selected_option = st.sidebar.empty()
    # for option in SIDEBAR_OPTIONS:
    #     if st.sidebar.button(option):
    #         selected_option = option

    # Set default selected option
    # selected_option = st.sidebar.selectbox("Select an option", SIDEBAR_OPTIONS)

    if selected_option == "Home":
        st.header("Welcome to AIOps! 👋")
        
        
        
# Your Streamlit app code goes here

    #elif selected_option == "VM Profiler":
        #profiling_dashboard()

    elif selected_option == "Memory Usage":
        memory_usage_dashboard()

    elif selected_option == "CPU Usage":
        cpu_usage_dashboard()

    elif selected_option == "Disk Usage":
        disk_usage_dashboard()

    elif selected_option == "Network Data":
        network_data_dashboard()
    
    

    

if not hasattr(st.session_state, "is_authenticated") or not st.session_state.is_authenticated:
    auth_app()
else:
    dashboard_app()
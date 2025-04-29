import streamlit as st
from displaymodelperformance import display_model_performance
from etflookup import etf_lookup
from calculatebeta import calculate_beta_page
from displaymodelgraphs import display_model_graphs

# Set the app title and logo
st.set_page_config(page_title="Foguth Financial Group", page_icon="🧮", layout="wide")
st.sidebar.image("assets/logo.png", use_container_width=True)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "Welcome to the Foguth ETP Model App",
    "Model Performance": display_model_performance,
    "ETF Lookup": etf_lookup,
    "Beta Calculator": calculate_beta_page,
    "Model Graphs": display_model_graphs,
}
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
if selected_page == "Home":
    st.title("Welcome to the Foguth Financial App")
    st.markdown(
        """
        This app provides tools for analyzing the Foguth ETP models, calculating betas, and visualizing model performance.
        
        **Features:**
        - View model performance metrics.
        - Lookup information on the individual ETFs.
        - Calculate weighted beta for each models and customize the time frame.
        - Visualize model graphs Year to Date.
        
        Use the navigation menu on the left to explore the app.
        """
    )
else:
    # Run the selected page's function
    pages[selected_page]()

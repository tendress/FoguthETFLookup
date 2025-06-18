import streamlit as st
from displaymodelperformance import display_model_performance
from etflookup import etf_lookup
from calculatebeta import calculate_beta_page
from displaymodelgraphs import display_model_graphs
from correlationmatrix import display_correlation_matrix
from economicindicators import economic_indicators
from consumerindicators import consumer_indicators
from livefactsheet import display_live_factsheet
import pandas as pd
import sqlite3
import datetime


# Set the app title and logo
st.set_page_config(page_title="Foguth Financial Group", page_icon="🧮", layout="wide")
st.sidebar.image("assets/logo.png", use_container_width=True)




# Sidebar navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "Welcome to Foguth ETP Model Insights",
    "Model Performance": display_model_performance,
    "Live ETP Factsheet": display_live_factsheet,
    "ETF Lookup": etf_lookup,
    "Beta Calculator": calculate_beta_page,
    "Correlation Matrix": display_correlation_matrix,
    "Economic Indicators": economic_indicators,
    "Consumer Indicators": consumer_indicators
    
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Display the last updated date
def get_last_updated_date():
    # Connect to the database
    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Fetch the last updated date from the database
    cursor.execute("SELECT MAX(updateDateTime) FROM ffgwebUpdateLog")
    last_updated = cursor.fetchone()[0]

    # Close the database connection
    conn.close()

    if last_updated:
        # Convert the last updated date to a datetime object
        last_updated_date = datetime.datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S')
        # Format the date with a time and AM/PM indicator
        formatted_date = last_updated_date.strftime('%B %d, %Y at %I:%M %p')
        return formatted_date

    else:
        return None
    
# Display the last updated date in the sidebar
last_updated_date = get_last_updated_date()
if last_updated_date:
    st.sidebar.markdown(f"**Last Updated:** {last_updated_date}")
    

# Display the selected page
if selected_page == "Home":
    # Database connection
    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Define the model groups
    model_groups = [
        ['Conservative Growth', 'Balanced Growth', 'Bullish Growth', 'Velocity', 'Opportunistic'],
        ['Conservative Value', 'Balanced Value', 'Bullish Value', 'Velocity', 'Opportunistic'],
        ['Rising Dividend Conservative', 'Rising Dividend Balanced', 'Rising Dividend Bullish', 'Rising Dividend Aggressive', 'Rising Dividend Momentum']
    ]

        # Fetch security sets and their weights for a given model
    @st.cache_data
    def fetch_security_sets_for_model(model_name):
        query = '''
            SELECT security_sets.name, model_security_set.weight
            FROM model_security_set
            JOIN security_sets ON model_security_set.security_set_id = security_sets.id
            JOIN models ON model_security_set.model_id = models.id
            WHERE models.name = ?
            ORDER BY security_sets.name ASC
        '''
        cursor.execute(query, (model_name,))
        results = cursor.fetchall()
        return [{"name": row[0], "weight": row[1] * 100} for row in results]

    # Fetch YTDPriceReturn for all models and store in a dictionary
    cursor.execute("SELECT name, YTDPriceReturn FROM models")
    ytd_returns = {row[0]: row[1] for row in cursor.fetchall()}

    # Initialize session state for toggling buttons
    if "open_buttons" not in st.session_state:
        st.session_state.open_buttons = {}

    
        
    # Display the model groups and their security sets
    st.title("ETP Model Menu")
    st.write("YTD Performance is updated periodically. Check the Model Performance tab to see Last Updated Date.")
    

    # Define group headings
    group_headings = ["Growth", "Value", "Rising Dividend"]

    # Iterate through each model group
    for group_index, (group, heading) in enumerate(zip(model_groups, group_headings)):
        # Add a heading for each group
        st.subheader(f"{heading} Models")

        # Create a row with columns for each model in the group
        cols = st.columns(len(group))
        for i, model in enumerate(group):
            with cols[i]:
                # Initialize the button state if not already set
                if model not in st.session_state.open_buttons:
                    st.session_state.open_buttons[model] = False

                # Display model name and YTD return
                ytd = ytd_returns.get(model, None)
                if ytd is not None:
                    st.markdown(f"<div style='text-align:center'><b>{model}</b><br><span style='font-size:18px;color:#0066CC;'>YTD: {ytd:.2f}%</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center'><b>{model}</b><br><span style='font-size:18px;color:#888;'>YTD: N/A</span></div>", unsafe_allow_html=True)

                

        # Add space and a horizontal line between each row of models
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Close the database connection
    conn.close()
    
    st.title("Welcome to Foguth ETP Model Insights")
    st.markdown(
            """
            This app provides tools for analyzing the Foguth ETP models, calculating betas, and visualizing model performance.
            
            Disclosure: The information provided in this app is for educational purposes only and should not be considered as financial advice. Performance results do not include advisory fees, which would reduce returns. Past performance is not indicative of future results. 
            
            ETP "Exchange Traded Portfolios"
            
            **Features:**
            - View model performance metrics.
            - Lookup information on the individual ETFs.
            - Calculate weighted beta for each models and customize the time frame.
            - Visualize model graphs Year to Date.
            
            Use the navigation menu on the left to explore the app.
            """
        )
    st.markdown(
        """
        <h1 style="color: red; font-size: 32px;">Sector Rotation Rebalance: Qualified 5/2 (Complete), Non-Qualified 5/5 (Complete)</h1>
        <ul style="color: red; font-size: 20px;">
            <li>XLE: The Energy Select Sector SPDR Fund - 25%</li>
            <li>XLU: The Utilities Select Sector SPDR Fund - 20%</li>
            <li>XLK: The Technology Select Sector SPDR Fund - 15%</li>
            <li>XLRE: The Real Estate Select Sector SPDR Fund - 15%</li>
            <li>XLF: The Financial Select Sector SPDR Fund - 15%</li>
            <li>XLC: The Communication Services Select Sector SPDR ETF Fund - 10%</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    
else:
# Run the selected page's function
    pages[selected_page]()

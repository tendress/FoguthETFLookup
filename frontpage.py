import streamlit as st
from displaymodelperformance import display_model_performance
from etflookup import etf_lookup
from calculatebeta import calculate_beta_page
from displaymodelgraphs import display_model_graphs
import sqlite3

# Set the app title and logo
st.set_page_config(page_title="Foguth Financial Group", page_icon="🧮", layout="wide")
st.sidebar.image("assets/logo.png", use_container_width=True)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "Welcome to Foguth ETP Model Insights",
    "Model Performance": display_model_performance,
    "Model Graphs": display_model_graphs,
    "ETF Lookup": etf_lookup,
    "Beta Calculator": calculate_beta_page,
    
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
if selected_page == "Home":
    st.title("Welcome to Foguth ETP Model Insights")
    st.markdown(
            """
            This app provides tools for analyzing the Foguth ETP models, calculating betas, and visualizing model performance.
            
            Disclosure: The information provided in this app is for educational purposes only and should not be considered as financial advice. Performance results do not include advisory fees, which would reduce returns. Past performance is not indicative of future results. Performance reflects the CURRENT makeup of the model and does not reflect the active management of the model. 
            
            ETP "Exchange Traded Portfolios"
            
            **Features:**
            - View model performance metrics.
            - Lookup information on the individual ETFs.
            - Calculate weighted beta for each models and customize the time frame.
            - Visualize model graphs Year to Date.
            
            Use the navigation menu on the left to explore the app.
            """
        )
    # Database connection
    database_path = 'foguth_etf_models.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Define the model groups
    model_groups = [
        ['Conservative Growth', 'Balanced Growth', 'Bullish Growth', 'Aggressive', 'Momentum'],
        ['Conservative Value', 'Balanced Value', 'Bullish Value', 'Aggressive', 'Momentum'],
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
        return [{"name": row[0], "weight": row[1] *100} for row in results]

    # Initialize session state for toggling buttons
    if "open_buttons" not in st.session_state:
        st.session_state.open_buttons = {}

    # Display the model groups and their security sets
    st.title("ETP Model Menu")
    st.write("Click on a model to view its associated security sets and their weights.")

    # Define group headings
    group_headings = ["Growth", "Value", "Rising Dividend"]

    # Iterate through each model group
    for group_index, (group, heading) in enumerate(zip(model_groups, group_headings)):
        # Add a heading for each group
        st.header(f"{heading} Models")

        # Create a row with columns for each model in the group
        cols = st.columns(len(group))
        for i, model in enumerate(group):
            with cols[i]:
                # Initialize the button state if not already set
                if model not in st.session_state.open_buttons:
                    st.session_state.open_buttons[model] = False

                # Create a button for each model with a unique key
                if st.button(model, key=f"{model}_{group_index}"):
                    # Toggle the button state
                    st.session_state.open_buttons[model] = not st.session_state.open_buttons[model]

                # Display security sets if the button is open
                if st.session_state.open_buttons[model]:
                    security_sets = fetch_security_sets_for_model(model)
                    if security_sets:
                        st.write(f"**Security Sets for {model}:**")
                        for security_set in security_sets:
                            st.write(f"- {security_set['name']} ({security_set['weight']}%)")
                    else:
                        st.write(f"No security sets found for {model}.")


    # Close the database connection
    conn.close()
else:
# Run the selected page's function
    pages[selected_page]()

### Foguth Financial ETF Lookup Tool ###
import yfinance as yf
import pandas as pd
import streamlit as st
import sqlite3

st.sidebar.image(
    "logo.png",
    caption="Foguth Financial Group",
    use_container_width=True
)


# Database connection
database_path = 'foguth_etf_models.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Fetch the list of ETF tickers from the database
cursor.execute('SELECT symbol FROM etfs')
myTickers = [row[0] for row in cursor.fetchall()]  # Create a list of tickers

# Function to fetch ETF data using yfinance
def etflookup(etf):
    etf_data = yf.Ticker(etf)
    return etf_data.get_funds_data()

# Cache ETF data and correlation matrix
@st.cache_data
def load_etf_data(myTickers):
    tickerdata = {}
    etf_corr = pd.DataFrame()

    for etf in myTickers:
        etf_info = etflookup(etf)
        top_holdings = etf_info.top_holdings
        top_holdings['Holding Percent'] = top_holdings['Holding Percent'] * 100  # Convert to percentage
        asset_classes = etf_info.asset_classes
        sector_weightings = etf_info.sector_weightings
        fund_overview = etf_info.fund_overview
        fund_operations = etf_info.fund_operations
        fund_description = etf_info.description

        etf_dict = {
            'Top Holdings': top_holdings,
            'Asset Classes': asset_classes,
            'Sector Weightings': sector_weightings,
            'Fund Overview': fund_overview,
            'Fund Operations': fund_operations,
            'Fund Description': fund_description
        }
        tickerdata[etf] = etf_dict

        etf_data = yf.Ticker(etf)
        etf_history = etf_data.history(period='max')
        etf_corr[etf] = etf_history['Close']

    etf_corr = etf_corr.pct_change().corr()
    return tickerdata, etf_corr

st.sidebar.title("Links")
st.sidebar.link_button(label="Foguth ETP Model Performance", url="https://foguthmodelperformance.streamlit.app/")

# Load ETF data and correlation matrix
tickerdata, etf_corr = load_etf_data(myTickers)

# Initialize session state for selected ETF
if 'selected_etf' not in st.session_state:
    st.session_state.selected_etf = myTickers[0]

# Sidebar: ETF selection
st.sidebar.title('Foguth Financial ETF Lookup Tool')
selected_etf = st.sidebar.selectbox(
    'Select an ETF',
    myTickers,
    key='etf_selectbox'
)

# Sidebar: Model selection
query = 'SELECT name FROM models'
cursor.execute(query)
models = cursor.fetchall()
foguthmodels = [model[0] for model in models]

if not foguthmodels:
    st.sidebar.warning("No models available in the database.")
else:
    selected_model = st.sidebar.selectbox(
        'Select a Model',
        foguthmodels,
        format_func=lambda x: x,
        key='model_selectbox'
    )

    # Query the database to get ETF weights grouped by security set
    query = '''
        SELECT 
            ss.name AS SecuritySet,
            e.symbol AS ETF,
            ms.weight * se.weight AS Weight
        FROM models m
        JOIN model_security_set ms ON m.id = ms.model_id
        JOIN security_sets ss ON ms.security_set_id = ss.id
        JOIN security_sets_etfs se ON ss.id = se.security_set_id
        JOIN etfs e ON se.etf_id = e.id
        WHERE m.name = ?
        ORDER BY SecuritySet, Weight DESC
    '''
    cursor.execute(query, (selected_model,))
    etf_weights = cursor.fetchall()

    # Display ETF weights grouped by security set on two lines
    st.sidebar.write("**ETF Weights by Strategy**")
    st.sidebar.write("**Model:**", selected_model)
    current_security_set = None
    for security_set, etf, weight in etf_weights:
        if security_set != current_security_set:
            st.sidebar.write(f"**{security_set}**")
            current_security_set = security_set
        st.sidebar.write(f"- {etf}: {weight:.2%}")

# Update session state for selected ETF
st.session_state.selected_etf = selected_etf

# Main content: Display selected ETF information
st.title('ETF Information')
st.header(selected_etf, divider=True)
st.write(tickerdata[selected_etf]['Fund Description'])
st.write(tickerdata[selected_etf]['Fund Overview'])

# Display top holdings
st.header('Top Holdings', divider=True)
st.write(tickerdata[selected_etf]['Top Holdings'])

# Display asset classes
st.header('Asset Classes', divider=True)
st.write(tickerdata[selected_etf]['Asset Classes'])

# Display sector weightings
st.header('Sector Weightings', divider=True)
st.write(tickerdata[selected_etf]['Sector Weightings'])

# Display fund operations
st.header('Fund Operations', divider=True)
st.write(tickerdata[selected_etf]['Fund Operations'])

# Display performance history graph
etf_data = yf.Ticker(selected_etf)
etf_history = etf_data.history(period='max')
st.header('Performance History', divider=True)
st.line_chart(etf_history['Close'])

# Display correlation matrix
st.header('Correlation Matrix', divider=True)
st.write(etf_corr.style.background_gradient(cmap='YlGn'))

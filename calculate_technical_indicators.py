"""
ETF Technical Indicators Analysis - Streamlit App
This app reads ETF price data and calculates/displays various technical indicators using TA-Lib.
"""

import sqlite3
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests

# Configure Streamlit page
st.set_page_config(page_title="Technical Indicators", page_icon="📊", layout="wide")

# Database configuration
SOURCE_DATABASE_PATH = 'foguth_etf_models.db'  # Read ETF data from here
INDICATORS_DATABASE_PATH = 'etf_technical_indicators.db'  # Store indicators here

def get_etf_list(conn):
    """
    Fetch all ETF symbols and IDs from the etfs table.
    
    Args:
        conn: SQLite database connection
        
    Returns:
        List of tuples (etf_id, symbol, name)
    """
    cursor = conn.cursor()
    cursor.execute('SELECT id, symbol, name FROM etfs ORDER BY symbol')
    return cursor.fetchall()

def get_etf_price_data(conn, etf_id, symbol, start_date=None, end_date=None, min_periods=200):
    """
    Fetch historical price data for a specific ETF.
    
    Args:
        conn: SQLite database connection
        etf_id: ETF ID from the etfs table
        symbol: ETF symbol
        start_date: Optional start date filter
        end_date: Optional end date filter
        min_periods: Minimum number of data points required
        
    Returns:
        DataFrame with Date and Close columns
    """
    if start_date and end_date:
        query = '''
            SELECT Date, Close, Volume 
            FROM etf_prices 
            WHERE etf_id = ? AND symbol = ? AND Date BETWEEN ? AND ?
            ORDER BY Date ASC
        '''
        df = pd.read_sql_query(query, conn, params=(etf_id, symbol, start_date, end_date))
    else:
        query = '''
            SELECT Date, Close, Volume 
            FROM etf_prices 
            WHERE etf_id = ? AND symbol = ?
            ORDER BY Date ASC
        '''
        df = pd.read_sql_query(query, conn, params=(etf_id, symbol))
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    if len(df) < min_periods:
        return None
    
    return df

def calculate_momentum_indicators(close_prices):
    """
    Calculate momentum-based technical indicators.
    
    Args:
        close_prices: numpy array of closing prices
        
    Returns:
        Dictionary of indicator values
    """
    indicators = {}
    
    try:
        # RSI - Relative Strength Index (14-day)
        indicators['RSI_14'] = talib.RSI(close_prices, timeperiod=14)
        
        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                                    fastperiod=12, 
                                                    slowperiod=26, 
                                                    signalperiod=9)
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Hist'] = macd_hist
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(close_prices, close_prices, close_prices,
                                    fastk_period=14,
                                    slowk_period=3,
                                    slowk_matype=0,
                                    slowd_period=3,
                                    slowd_matype=0)
        indicators['STOCH_K'] = slowk
        indicators['STOCH_D'] = slowd
        
        # Commodity Channel Index
        indicators['CCI_14'] = talib.CCI(close_prices, close_prices, close_prices, timeperiod=14)
        
        # Williams %R
        indicators['WILLR_14'] = talib.WILLR(close_prices, close_prices, close_prices, timeperiod=14)
        
        # Rate of Change
        indicators['ROC_10'] = talib.ROC(close_prices, timeperiod=10)
        
        # Momentum
        indicators['MOM_10'] = talib.MOM(close_prices, timeperiod=10)
        
    except Exception as e:
        print(f"Error calculating momentum indicators: {e}")
    
    return indicators

def calculate_trend_indicators(close_prices):
    """
    Calculate trend-based technical indicators.
    
    Args:
        close_prices: numpy array of closing prices
        
    Returns:
        Dictionary of indicator values
    """
    indicators = {}
    
    try:
        # Simple Moving Averages
        indicators['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        indicators['SMA_100'] = talib.SMA(close_prices, timeperiod=100)
        indicators['SMA_200'] = talib.SMA(close_prices, timeperiod=200)
        
        # Exponential Moving Averages
        indicators['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
        indicators['EMA_26'] = talib.EMA(close_prices, timeperiod=26)
        indicators['EMA_50'] = talib.EMA(close_prices, timeperiod=50)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, 
                                             timeperiod=20,
                                             nbdevup=2,
                                             nbdevdn=2,
                                             matype=0)
        indicators['BB_Upper'] = upper
        indicators['BB_Middle'] = middle
        indicators['BB_Lower'] = lower
        
        # Bollinger Band Width (BBW) - indicates squeeze
        # BBW = (Upper Band - Lower Band) / Middle Band
        indicators['BBW'] = (upper - lower) / middle
        
        # Average Directional Index (ADX)
        indicators['ADX_14'] = talib.ADX(close_prices, close_prices, close_prices, timeperiod=14)
        
        # Parabolic SAR
        indicators['SAR'] = talib.SAR(close_prices, close_prices, acceleration=0.02, maximum=0.2)
        
        # Average True Range
        indicators['ATR_14'] = talib.ATR(close_prices, close_prices, close_prices, timeperiod=14)
        
    except Exception as e:
        print(f"Error calculating trend indicators: {e}")
    
    return indicators

def calculate_volume_indicators(close_prices, volume=None):
    """
    Calculate volume-based technical indicators.
    
    Args:
        close_prices: numpy array of closing prices
        volume: numpy array of volume data (if available)
        
    Returns:
        Dictionary of indicator values
    """
    indicators = {}
    
    # Check if volume data is valid
    has_valid_volume = volume is not None and len(volume) > 0 and not np.all(np.isnan(volume))
    
    if has_valid_volume:
        try:
            # Convert volume to float64 (double) to ensure compatibility with talib
            volume_float = np.asarray(volume, dtype=np.float64)
            close_float = np.asarray(close_prices, dtype=np.float64)
            
            # On-Balance Volume
            indicators['OBV'] = talib.OBV(close_float, volume_float)
            
            # Accumulation/Distribution Line
            indicators['AD'] = talib.AD(close_float, close_float, close_float, volume_float)
            
            # Chaikin A/D Oscillator
            indicators['ADOSC'] = talib.ADOSC(close_float, close_float, close_float, volume_float,
                                              fastperiod=3, slowperiod=10)
            
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
    
    return indicators

def calculate_all_indicators(etf_id, symbol, df):
    """
    Calculate all technical indicators for a given ETF.
    
    Args:
        etf_id: ETF ID
        symbol: ETF symbol
        df: DataFrame with Date, Close, and optionally Volume columns
        
    Returns:
        DataFrame with all indicators
    """
    close_prices = df['Close'].values
    
    # Get volume data if available
    volume_data = df['Volume'].values if 'Volume' in df.columns else None
    
    # Calculate all indicator groups
    momentum = calculate_momentum_indicators(close_prices)
    trend = calculate_trend_indicators(close_prices)
    volume = calculate_volume_indicators(close_prices, volume_data)
    
    # Combine all indicators into the dataframe
    result_df = df.copy()
    result_df['etf_id'] = etf_id
    result_df['symbol'] = symbol
    
    # Add all momentum indicators
    for indicator_name, values in momentum.items():
        result_df[indicator_name] = values
    
    # Add all trend indicators
    for indicator_name, values in trend.items():
        result_df[indicator_name] = values
    
    # Add all volume indicators
    for indicator_name, values in volume.items():
        result_df[indicator_name] = values
    
    return result_df

def get_or_create_indicators(etf_id, symbol, force_recalculate=False):
    """
    Get indicators from database or calculate if not exists.
    
    Args:
        etf_id: ETF ID
        symbol: ETF symbol
        force_recalculate: If True, recalculate even if data exists
        
    Returns:
        DataFrame with indicators
    """
    source_conn = sqlite3.connect(SOURCE_DATABASE_PATH)
    indicators_conn = sqlite3.connect(INDICATORS_DATABASE_PATH)
    
    try:
        # Check if indicators already exist
        if not force_recalculate:
            try:
                existing = pd.read_sql_query(
                    'SELECT * FROM etf_technical_indicators WHERE etf_id = ? ORDER BY Date',
                    indicators_conn,
                    params=(etf_id,)
                )
                if len(existing) > 0:
                    existing['Date'] = pd.to_datetime(existing['Date'])
                    return existing
            except:
                pass  # Table doesn't exist yet
        
        # Calculate indicators
        df = get_etf_price_data(source_conn, etf_id, symbol)
        
        if df is None or len(df) < 200:
            return None
        
        indicators_df = calculate_all_indicators(etf_id, symbol, df)
        
        # Save to indicators database
        save_indicators_to_database(indicators_conn, indicators_df, replace=force_recalculate)
        
        return indicators_df
        
    finally:
        source_conn.close()
        indicators_conn.close()

def create_indicators_table(conn):
    """
    Create a table to store technical indicators in the database.
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS etf_technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etf_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            Date TEXT NOT NULL,
            Close REAL,
            RSI_14 REAL,
            MACD REAL,
            MACD_Signal REAL,
            MACD_Hist REAL,
            STOCH_K REAL,
            STOCH_D REAL,
            CCI_14 REAL,
            WILLR_14 REAL,
            ROC_10 REAL,
            MOM_10 REAL,
            SMA_20 REAL,
            SMA_50 REAL,
            SMA_100 REAL,
            SMA_200 REAL,
            EMA_12 REAL,
            EMA_26 REAL,
            EMA_50 REAL,
            BB_Upper REAL,
            BB_Middle REAL,
            BB_Lower REAL,
            BBW REAL,
            ADX_14 REAL,
            SAR REAL,
            ATR_14 REAL,
            OBV REAL,
            AD REAL,
            ADOSC REAL,
            UNIQUE(etf_id, Date)
        )
    ''')
    
    conn.commit()

def save_indicators_to_database(conn, df, replace=False):
    """
    Save calculated indicators to the database.
    
    Args:
        conn: SQLite database connection
        df: DataFrame with indicators
        replace: If True, delete existing data for this ETF first
    """
    # Convert Date to string format
    df_to_save = df.copy()
    df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d')
    
    # Replace NaN with None for database insertion
    df_to_save = df_to_save.replace({np.nan: None})
    
    if replace and 'etf_id' in df_to_save.columns:
        etf_id = df_to_save['etf_id'].iloc[0]
        cursor = conn.cursor()
        cursor.execute('DELETE FROM etf_technical_indicators WHERE etf_id = ?', (etf_id,))
        conn.commit()
    
    # Save to database
    df_to_save.to_sql('etf_technical_indicators', conn, if_exists='append', index=False)

def plot_price_with_indicators(df, symbol, indicator_type, selected_ma=None):
    """
    Create interactive plots showing price and technical indicators.
    
    Args:
        df: DataFrame with price and indicators
        symbol: ETF symbol
        indicator_type: Type of indicators to display
        selected_ma: List of selected moving averages to display (optional)
    """
    if indicator_type == "Moving Averages":
        # Create figure with secondary y-axis
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        # Add price line
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)))
        
        # Add moving averages
        ma_indicators = ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'EMA_50']
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta']
        
        # Filter to only selected MAs if provided
        if selected_ma:
            ma_indicators = [ma for ma in ma_indicators if ma in selected_ma]
        
        for ma, color in zip(ma_indicators, colors):
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[ma], name=ma,
                                        line=dict(color=color, width=1.5)))
        
        fig.update_layout(
            title=f"{symbol} - Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=600
        )
        
    elif indicator_type == "Bollinger Bands":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3],
                           subplot_titles=(f"{symbol} - Price with Bollinger Bands", "Bollinger Band Width (Squeeze Indicator)"))
        
        # Add Bollinger Bands on top subplot
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                                line=dict(color='red', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Middle'], name='BB Middle',
                                line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                                line=dict(color='red', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(255,0,0,0.1)'), row=1, col=1)
        
        # Add price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)), row=1, col=1)
        
        # Add Bollinger Band Width on bottom subplot
        if 'BBW' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BBW'], name='BB Width',
                                    line=dict(color='purple', width=2),
                                    fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'), row=2, col=1)
            
            # Calculate 6-month (126 trading days) rolling minimum for squeeze detection
            df_with_rolling = df.copy()
            df_with_rolling['BBW_6M_Min'] = df_with_rolling['BBW'].rolling(window=126, min_periods=1).min()
            
            # Add 6-month minimum line
            fig.add_trace(go.Scatter(x=df_with_rolling['Date'], y=df_with_rolling['BBW_6M_Min'], 
                                    name='6-Month Low',
                                    line=dict(color='orange', width=1, dash='dash')), row=2, col=1)
            
            # Identify squeeze points (when BBW equals 6-month minimum)
            # Allow small tolerance for floating point comparison
            squeeze_threshold = 1.001  # 0.1% tolerance
            df_with_rolling['Is_Squeeze'] = (df_with_rolling['BBW'] / df_with_rolling['BBW_6M_Min']) < squeeze_threshold
            
            # Mark squeeze periods with red background shading
            squeeze_points = df_with_rolling[df_with_rolling['Is_Squeeze']]
            if not squeeze_points.empty:
                for idx in squeeze_points.index:
                    fig.add_vrect(
                        x0=df_with_rolling.loc[idx, 'Date'],
                        x1=df_with_rolling.loc[idx, 'Date'],
                        fillcolor="red", opacity=0.1,
                        layer="below", line_width=0,
                        row=2, col=1
                    )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="BBW", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - Bollinger Bands with Squeeze Indicator",
            hovermode='x unified',
            height=700
        )
        
    elif indicator_type == "RSI":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Add price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)), row=1, col=1)
        
        # Add RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], name='RSI (14)',
                                line=dict(color='purple', width=2)), row=2, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - RSI Indicator",
            hovermode='x unified',
            height=700
        )
        
    elif indicator_type == "MACD":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Add price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)), row=1, col=1)
        
        # Add MACD
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                                line=dict(color='blue', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
                                line=dict(color='red', width=2)), row=2, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram',
                            marker_color='gray'), row=2, col=1)
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - MACD Indicator",
            hovermode='x unified',
            height=700
        )
        
    elif indicator_type == "Stochastic":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Add price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)), row=1, col=1)
        
        # Add Stochastic
        fig.add_trace(go.Scatter(x=df['Date'], y=df['STOCH_K'], name='%K',
                                line=dict(color='blue', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['STOCH_D'], name='%D',
                                line=dict(color='red', width=2)), row=2, col=1)
        
        # Add levels
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Stochastic", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - Stochastic Oscillator",
            hovermode='x unified',
            height=700
        )
    
    elif indicator_type == "OBV":
        if 'OBV' in df.columns and df['OBV'].notna().any():
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Add price
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                    line=dict(color='black', width=2)), row=1, col=1)
            
            # Add OBV
            fig.add_trace(go.Scatter(x=df['Date'], y=df['OBV'], name='OBV',
                                    line=dict(color='blue', width=2), fill='tozeroy',
                                    fillcolor='rgba(0,100,255,0.2)'), row=2, col=1)
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="On-Balance Volume", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            fig.update_layout(
                title=f"{symbol} - On-Balance Volume (OBV)",
                hovermode='x unified',
                height=700
            )
        else:
            # No volume data available
            fig = go.Figure()
            fig.add_annotation(
                text="OBV indicator requires volume data, which is not available for this ticker",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(height=400)
    
    elif indicator_type == "ADX":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Add price
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price',
                                line=dict(color='black', width=2)), row=1, col=1)
        
        # Add ADX
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX_14'], name='ADX (14)',
                                line=dict(color='purple', width=2)), row=2, col=1)
        
        # Add trend strength levels
        fig.add_hline(y=25, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Strong Trend")
        fig.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=1, annotation_text="Moderate")
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="ADX", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"{symbol} - Average Directional Index (ADX)",
            hovermode='x unified',
            height=700
        )
    
    return fig

def analyze_buy_sell_signals(df):
    """
    Analyze technical indicators to generate buy/sell signals.
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        Dictionary with signal analysis
    """
    latest = df.iloc[-1]
    signals = []
    buy_count = 0
    sell_count = 0
    neutral_count = 0
    
    # RSI Analysis
    if pd.notna(latest['RSI_14']):
        if latest['RSI_14'] < 30:
            signals.append(("RSI (14)", "BUY", f"Oversold at {latest['RSI_14']:.2f}"))
            buy_count += 1
        elif latest['RSI_14'] > 70:
            signals.append(("RSI (14)", "SELL", f"Overbought at {latest['RSI_14']:.2f}"))
            sell_count += 1
        else:
            signals.append(("RSI (14)", "NEUTRAL", f"Neutral at {latest['RSI_14']:.2f}"))
            neutral_count += 1
    
    # MACD Analysis
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            signals.append(("MACD", "BUY", "Bullish crossover"))
            buy_count += 1
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            signals.append(("MACD", "SELL", "Bearish crossover"))
            sell_count += 1
        else:
            signals.append(("MACD", "NEUTRAL", "No clear signal"))
            neutral_count += 1
    
    # Moving Average Analysis - Price vs SMA 50
    if pd.notna(latest['SMA_50']):
        if latest['Close'] > latest['SMA_50']:
            signals.append(("Price vs SMA50", "BUY", f"Price above SMA50"))
            buy_count += 1
        else:
            signals.append(("Price vs SMA50", "SELL", f"Price below SMA50"))
            sell_count += 1
    
    # Moving Average Analysis - Price vs SMA 200
    if pd.notna(latest['SMA_200']):
        if latest['Close'] > latest['SMA_200']:
            signals.append(("Price vs SMA200", "BUY", f"Price above SMA200"))
            buy_count += 1
        else:
            signals.append(("Price vs SMA200", "SELL", f"Price below SMA200"))
            sell_count += 1
    
    # Golden/Death Cross Analysis
    if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        if latest['SMA_50'] > latest['SMA_200']:
            signals.append(("SMA Cross", "BUY", "Golden Cross (SMA50 > SMA200)"))
            buy_count += 1
        else:
            signals.append(("SMA Cross", "SELL", "Death Cross (SMA50 < SMA200)"))
            sell_count += 1
    
    # Stochastic Analysis
    if pd.notna(latest['STOCH_K']):
        if latest['STOCH_K'] < 20:
            signals.append(("Stochastic", "BUY", f"Oversold at {latest['STOCH_K']:.2f}"))
            buy_count += 1
        elif latest['STOCH_K'] > 80:
            signals.append(("Stochastic", "SELL", f"Overbought at {latest['STOCH_K']:.2f}"))
            sell_count += 1
        else:
            signals.append(("Stochastic", "NEUTRAL", f"Neutral at {latest['STOCH_K']:.2f}"))
            neutral_count += 1
    
    # Bollinger Band Squeeze Analysis
    if 'BBW' in df.columns and pd.notna(latest['BBW']) and len(df) >= 126:
        # Calculate 6-month (126 trading days) minimum BBW
        bbw_6m_min = df['BBW'].tail(126).min()
        
        # Calculate 6-month minimum price
        price_6m_min = df['Close'].tail(126).min()
        
        # Check if in squeeze (BBW at 6-month low with 0.1% tolerance)
        is_squeeze = (latest['BBW'] / bbw_6m_min) < 1.001
        
        # Check if price is within 5% of 6-month low
        price_near_low = (latest['Close'] - price_6m_min) / price_6m_min <= 0.05
        
        if is_squeeze and price_near_low:
            price_pct_above_low = ((latest['Close'] - price_6m_min) / price_6m_min) * 100
            signals.append(("BB Squeeze", "BUY", f"Squeeze at 6M low, price {price_pct_above_low:.1f}% above 6M low"))
            buy_count += 1
        elif is_squeeze:
            signals.append(("BB Squeeze", "NEUTRAL", "Squeeze detected, price not near 6M low"))
            neutral_count += 1
        else:
            signals.append(("BB Squeeze", "NEUTRAL", "No squeeze detected"))
            neutral_count += 1
    
    # ADX Trend Strength
    trend_strength = "N/A"
    if pd.notna(latest['ADX_14']):
        if latest['ADX_14'] > 25:
            trend_strength = "Strong"
        elif latest['ADX_14'] > 20:
            trend_strength = "Moderate"
        else:
            trend_strength = "Weak"
    
    # Overall Signal
    total_signals = buy_count + sell_count + neutral_count
    if total_signals > 0:
        buy_percentage = (buy_count / total_signals) * 100
        sell_percentage = (sell_count / total_signals) * 100
        
        if buy_percentage >= 60:
            overall_signal = "BUY"
        elif sell_percentage >= 60:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"
    else:
        overall_signal = "INSUFFICIENT DATA"
        buy_percentage = 0
        sell_percentage = 0
    
    return {
        'overall': overall_signal,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'neutral_count': neutral_count,
        'buy_percentage': buy_percentage,
        'sell_percentage': sell_percentage,
        'signals': signals,
        'trend_strength': trend_strength
    }

def display_buy_sell_signal(signal_analysis, symbol):
    """
    Display buy/sell signal prominently.
    
    Args:
        signal_analysis: Dictionary from analyze_buy_sell_signals
        symbol: ETF symbol
    """
    overall = signal_analysis['overall']
    
    # Color coding
    if overall == "BUY":
        color = "#00CC00"
        icon = "🟢"
        recommendation = "Consider Buying"
    elif overall == "SELL":
        color = "#CC0000"
        icon = "🔴"
        recommendation = "Consider Selling"
    else:
        color = "#FFA500"
        icon = "🟡"
        recommendation = "Hold / Wait"
    
    # Display signal box
    st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        ">
            <h1 style="color: white; margin: 0;">{icon} {overall} SIGNAL</h1>
            <h3 style="color: white; margin: 10px 0;">{recommendation} - {symbol}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Display signal breakdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Buy Signals", signal_analysis['buy_count'], 
                 delta=f"{signal_analysis['buy_percentage']:.0f}%")
    
    with col2:
        st.metric("Sell Signals", signal_analysis['sell_count'],
                 delta=f"-{signal_analysis['sell_percentage']:.0f}%", delta_color="inverse")
    
    with col3:
        st.metric("Neutral Signals", signal_analysis['neutral_count'])
    
    with col4:
        st.metric("Trend Strength", signal_analysis['trend_strength'])
    
    # Show detailed signals
    with st.expander("📊 View Detailed Signal Breakdown"):
        for indicator, signal, description in signal_analysis['signals']:
            if signal == "BUY":
                st.success(f"**{indicator}**: {signal} - {description}")
            elif signal == "SELL":
                st.error(f"**{indicator}**: {signal} - {description}")
            else:
                st.info(f"**{indicator}**: {signal} - {description}")

def display_indicator_values(df, symbol):
    """
    Display current values of technical indicators.
    
    Args:
        df: DataFrame with indicators
        symbol: ETF symbol
    """
    latest = df.iloc[-1]
    
    st.subheader(f"Current Indicator Values for {symbol}")
    st.caption(f"As of {latest['Date'].strftime('%Y-%m-%d')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Close Price", f"${latest['Close']:.2f}")
        st.metric("RSI (14)", f"{latest['RSI_14']:.2f}" if pd.notna(latest['RSI_14']) else "N/A")
        st.metric("ADX (14)", f"{latest['ADX_14']:.2f}" if pd.notna(latest['ADX_14']) else "N/A")
    
    with col2:
        st.metric("SMA (50)", f"${latest['SMA_50']:.2f}" if pd.notna(latest['SMA_50']) else "N/A")
        st.metric("SMA (200)", f"${latest['SMA_200']:.2f}" if pd.notna(latest['SMA_200']) else "N/A")
        st.metric("EMA (50)", f"${latest['EMA_50']:.2f}" if pd.notna(latest['EMA_50']) else "N/A")
    
    with col3:
        st.metric("MACD", f"{latest['MACD']:.3f}" if pd.notna(latest['MACD']) else "N/A")
        st.metric("MACD Signal", f"{latest['MACD_Signal']:.3f}" if pd.notna(latest['MACD_Signal']) else "N/A")
        st.metric("MACD Hist", f"{latest['MACD_Hist']:.3f}" if pd.notna(latest['MACD_Hist']) else "N/A")
    
    with col4:
        st.metric("Stoch %K", f"{latest['STOCH_K']:.2f}" if pd.notna(latest['STOCH_K']) else "N/A")
        st.metric("Stoch %D", f"{latest['STOCH_D']:.2f}" if pd.notna(latest['STOCH_D']) else "N/A")
        st.metric("OBV", f"{latest['OBV']:,.0f}" if pd.notna(latest.get('OBV')) else "N/A")
    
    # Add fundamental metrics section
    st.markdown("---")
    st.markdown("#### 📊 Key Fundamental Metrics")
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pe_ratio = info.get('trailingPE', None)
            if pe_ratio is not None:
                st.metric("P/E Ratio", f"{pe_ratio:.2f}")
            else:
                st.metric("P/E Ratio", "N/A")
        
        with col2:
            # Try multiple possible keys for PEG ratio
            peg_ratio = info.get('pegRatio') or info.get('trailingPegRatio') or info.get('forwardPegRatio')
            if peg_ratio is not None:
                st.metric("PEG Ratio", f"{peg_ratio:.2f}")
            else:
                st.metric("PEG Ratio", "N/A")
                # Debug: show if any PEG-related keys exist
                peg_keys = [k for k in info.keys() if 'peg' in k.lower()]
                if peg_keys:
                    st.caption(f"Debug: Found keys: {', '.join(peg_keys)}")
        
        with col3:
            pb_ratio = info.get('priceToBook', None)
            if pb_ratio is not None:
                st.metric("P/B Ratio", f"{pb_ratio:.2f}")
            else:
                st.metric("P/B Ratio", "N/A")
        
        with col4:
            div_yield = info.get('dividendYield', None)
            if div_yield is not None:
                st.metric("Dividend Yield", f"{div_yield:.2f}%")
            else:
                st.metric("Dividend Yield", "N/A")
    
    except Exception as e:
        st.caption(f"Unable to fetch fundamental data: {str(e)}")

def get_all_etf_signals(etf_list):
    """
    Get buy/sell signals for all ETFs.
    
    Args:
        etf_list: List of (etf_id, symbol, name) tuples
        
    Returns:
        DataFrame with signal analysis for all ETFs
    """
    signal_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (etf_id, symbol, name) in enumerate(etf_list):
        status_text.text(f"Analyzing {symbol}... ({idx+1}/{len(etf_list)})")
        progress_bar.progress((idx + 1) / len(etf_list))
        
        try:
            df = get_or_create_indicators(etf_id, symbol, force_recalculate=False)
            
            if df is not None and len(df) > 0:
                signal_analysis = analyze_buy_sell_signals(df)
                
                # Create a dictionary with individual indicator signals
                row_data = {
                    'Symbol': symbol,
                    'Name': name,
                    'Signal': signal_analysis['overall'],
                    'Buy Count': signal_analysis['buy_count'],
                    'Sell Count': signal_analysis['sell_count'],
                    'Neutral Count': signal_analysis['neutral_count'],
                    'Buy %': signal_analysis['buy_percentage'],
                    'Sell %': signal_analysis['sell_percentage'],
                    'Trend': signal_analysis['trend_strength'],
                    'Close': df.iloc[-1]['Close']
                }
                
                # Add individual indicator signals and their detailed reasons
                for indicator, signal, reason in signal_analysis['signals']:
                    row_data[indicator] = signal
                    row_data[f"{indicator} Detail"] = f"{signal} - {reason}"
                
                signal_data.append(row_data)
        except Exception as e:
            # Skip ETFs with errors
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(signal_data)

def plot_all_signals_chart(signals_df):
    """
    Create a color-coded chart showing buy/sell signals for all ETFs.
    
    Args:
        signals_df: DataFrame with signal analysis
    """
    if signals_df.empty:
        st.warning("No signal data available to display.")
        return
    
    # Sort by signal type and buy percentage
    signals_df['Signal_Order'] = signals_df['Signal'].map({'BUY': 0, 'NEUTRAL': 1, 'SELL': 2})
    signals_df = signals_df.sort_values(['Signal_Order', 'Buy %'], ascending=[True, False])
    
    # Calculate Neutral % (to make bars add up to 100%)
    signals_df['Neutral %'] = 100 - signals_df['Buy %'] - signals_df['Sell %']
    
    # Create color map
    color_map = {
        'BUY': '#00CC00',
        'SELL': '#CC0000',
        'NEUTRAL': '#FFA500'
    }
    signals_df['Color'] = signals_df['Signal'].map(color_map)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=signals_df['Symbol'],
        x=signals_df['Buy %'],
        name='Buy Signals',
        orientation='h',
        marker=dict(color='#00CC00'),
        text=signals_df['Buy %'].round(1),
        texttemplate='%{text}%',
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Buy Signals: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=signals_df['Symbol'],
        x=signals_df['Neutral %'],
        name='Neutral Signals',
        orientation='h',
        marker=dict(color='#0080FF'),
        text=signals_df['Neutral %'].round(1),
        texttemplate='%{text}%',
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Neutral Signals: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=signals_df['Symbol'],
        x=signals_df['Sell %'],
        name='Sell Signals',
        orientation='h',
        marker=dict(color='#CC0000'),
        text=signals_df['Sell %'].round(1),
        texttemplate='%{text}%',
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Sell Signals: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Buy vs Sell vs Neutral Signals Across All ETFs',
        xaxis_title='Percentage of Signals',
        yaxis_title='ETF Symbol',
        barmode='stack',
        height=max(600, len(signals_df) * 25),
        showlegend=True,
        hovermode='y unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create summary table with color coding
    st.subheader("Signal Summary Table")
    
    # Style the dataframe
    def highlight_signal(row):
        if row['Signal'] == 'BUY':
            return ['background-color: #00CC0033'] * len(row)
        elif row['Signal'] == 'SELL':
            return ['background-color: #CC000033'] * len(row)
        else:
            return ['background-color: #FFA50033'] * len(row)
    
    display_df = signals_df[['Symbol', 'Name', 'Signal', 'Buy Count', 'Sell Count', 
                             'Neutral Count', 'Buy %', 'Sell %', 'Trend', 'Close']].copy()
    display_df['Buy %'] = display_df['Buy %'].round(1)
    display_df['Sell %'] = display_df['Sell %'].round(1)
    display_df['Close'] = display_df['Close'].round(2)
    
    styled_df = display_df.style.apply(highlight_signal, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Summary statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    buy_count = len(signals_df[signals_df['Signal'] == 'BUY'])
    sell_count = len(signals_df[signals_df['Signal'] == 'SELL'])
    neutral_count = len(signals_df[signals_df['Signal'] == 'NEUTRAL'])
    
    with col1:
        st.metric("ETFs with BUY Signal", buy_count, delta=f"{(buy_count/len(signals_df)*100):.1f}%")
    with col2:
        st.metric("ETFs with SELL Signal", sell_count, delta=f"-{(sell_count/len(signals_df)*100):.1f}%", delta_color="inverse")
    with col3:
        st.metric("ETFs with NEUTRAL Signal", neutral_count)

def download_ticker_data(ticker_symbol, period="5y"):
    """
    Download historical data for any ticker from yfinance.
    
    Args:
        ticker_symbol: Ticker symbol to download
        period: Period of data to download (default: 5y)
        
    Returns:
        DataFrame with Date and Close columns
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return None, "No data found for this ticker symbol."
        
        # Create dataframe with Date, Close, and Volume
        df = pd.DataFrame({
            'Date': hist.index,
            'Close': hist['Close'].values,
            'Volume': hist['Volume'].values if 'Volume' in hist.columns else None
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.reset_index(drop=True)
        
        return df, None
        
    except Exception as e:
        return None, f"Error downloading data: {str(e)}"

def analyze_custom_ticker(ticker_symbol):
    """
    Download and analyze a custom ticker symbol.
    
    Args:
        ticker_symbol: Ticker symbol to analyze
        
    Returns:
        DataFrame with indicators or None if error
    """
    # Download data
    with st.spinner(f"Downloading data for {ticker_symbol}..."):
        df, error = download_ticker_data(ticker_symbol.upper())
    
    if error:
        st.error(error)
        return None
    
    if len(df) < 200:
        st.error(f"Insufficient data for {ticker_symbol}. Found {len(df)} data points, need at least 200.")
        return None
    
    # Calculate indicators
    with st.spinner(f"Calculating technical indicators for {ticker_symbol}..."):
        indicators_df = calculate_all_indicators(0, ticker_symbol.upper(), df)
    
    return indicators_df

def get_sp500_tickers():
    """
    Get list of S&P 500 ticker symbols from Wikipedia.
    
    Returns:
        List of ticker symbols
    """
    try:
        # Get S&P 500 list from Wikipedia with proper headers to avoid 403 error
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # Add headers to avoid 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        tables = pd.read_html(url, storage_options=headers)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean up tickers (replace dots with dashes for yfinance)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        return tickers
    except Exception as e:
        error_msg = str(e)
        if 'lxml' in error_msg.lower():
            st.error("""
            **Missing Required Package: lxml**
            
            Please install lxml by running this command in your terminal:
            ```
            pip install lxml
            ```
            
            Then **restart the Streamlit app** for the changes to take effect.
            """)
        else:
            st.error(f"Error fetching S&P 500 list: {error_msg}")
        return []

def download_sp500_data(tickers, period="2y"):
    """
    Download historical data for all S&P 500 tickers in batch.
    
    Args:
        tickers: List of ticker symbols
        period: Period of data to download (default: 2y)
        
    Returns:
        Dictionary mapping ticker to DataFrame with Date and Close columns
    """
    ticker_data = {}
    
    try:
        # Download all tickers in batch (much faster than individual downloads)
        st.info(f"Downloading data for {len(tickers)} S&P 500 stocks...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download in chunks to avoid timeouts
        chunk_size = 50
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            progress_bar.progress(min((i + chunk_size) / len(tickers), 1.0))
            status_text.text(f"Downloading chunk {i//chunk_size + 1}/{(len(tickers)//chunk_size) + 1}...")
            
            try:
                # Download chunk with error handling
                data = yf.download(chunk, period=period, group_by='ticker', progress=False, threads=True)
                
                if data.empty:
                    continue
                
                # Handle single vs multiple tickers
                if len(chunk) == 1:
                    # Single ticker - data is not multi-indexed
                    ticker = chunk[0]
                    if not data.empty and 'Close' in data.columns:
                        df = pd.DataFrame({
                            'Date': data.index,
                            'Close': data['Close'].values,
                            'Volume': data['Volume'].values if 'Volume' in data.columns else None
                        })
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.dropna(subset=['Close'])  # Only drop rows with missing Close prices
                        df = df.reset_index(drop=True)
                        
                        if len(df) >= 200:
                            ticker_data[ticker] = df
                else:
                    # Multiple tickers - data is multi-indexed
                    for ticker in chunk:
                        try:
                            # Check if ticker exists in the multi-level columns
                            if ticker in data.columns.get_level_values(0):
                                ticker_df = data[ticker]
                                
                                if not ticker_df.empty and 'Close' in ticker_df.columns:
                                    df = pd.DataFrame({
                                        'Date': ticker_df.index,
                                        'Close': ticker_df['Close'].values,
                                        'Volume': ticker_df['Volume'].values if 'Volume' in ticker_df.columns else None
                                    })
                                    df['Date'] = pd.to_datetime(df['Date'])
                                    df = df.dropna(subset=['Close'])  # Only drop rows with missing Close prices
                                    df = df.reset_index(drop=True)
                                    
                                    if len(df) >= 200:
                                        ticker_data[ticker] = df
                        except Exception as e:
                            # Skip this ticker and continue
                            continue
                            
            except Exception as e:
                # Skip this chunk and continue with next
                st.warning(f"Failed to download chunk starting at position {i}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if ticker_data:
            st.success(f"Successfully downloaded data for {len(ticker_data)} stocks")
        else:
            st.error("No data was successfully downloaded. Please check your internet connection and try again.")
        
        return ticker_data
        
    except Exception as e:
        st.error(f"Error downloading S&P 500 data: {str(e)}")
        return {}

def detect_macd_crossover(df, lookback_days=5):
    """
    Detect MACD crossovers in the past N days.
    
    Args:
        df: DataFrame with MACD indicators
        lookback_days: Number of days to look back for crossovers
        
    Returns:
        Dictionary with crossover info or None if no crossover
    """
    if len(df) < lookback_days + 1:
        return None
    
    # Get the last N days
    recent_df = df.tail(lookback_days + 1).copy()
    
    if 'MACD' not in recent_df.columns or 'MACD_Signal' not in recent_df.columns:
        return None
    
    # Check for crossovers
    for i in range(1, len(recent_df)):
        prev_idx = recent_df.index[i-1]
        curr_idx = recent_df.index[i]
        
        prev_macd = recent_df.loc[prev_idx, 'MACD']
        prev_signal = recent_df.loc[prev_idx, 'MACD_Signal']
        curr_macd = recent_df.loc[curr_idx, 'MACD']
        curr_signal = recent_df.loc[curr_idx, 'MACD_Signal']
        
        # Skip if any values are NaN
        if pd.isna(prev_macd) or pd.isna(prev_signal) or pd.isna(curr_macd) or pd.isna(curr_signal):
            continue
        
        # Bullish crossover: MACD crosses above Signal
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            return {
                'type': 'BUY',
                'date': recent_df.loc[curr_idx, 'Date'],
                'days_ago': len(recent_df) - i - 1,
                'macd': curr_macd,
                'signal': curr_signal,
                'current_position': 'MACD > Signal'
            }
        
        # Bearish crossover: MACD crosses below Signal
        if prev_macd >= prev_signal and curr_macd < curr_signal:
            return {
                'type': 'SELL',
                'date': recent_df.loc[curr_idx, 'Date'],
                'days_ago': len(recent_df) - i - 1,
                'macd': curr_macd,
                'signal': curr_signal,
                'current_position': 'Signal > MACD'
            }
    
    return None

def analyze_sp500_signals():
    """
    Analyze buy/sell signals for all S&P 500 stocks.
    
    Returns:
        DataFrame with signal analysis for all S&P 500 stocks
    """
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    
    if not tickers:
        return None
    
    st.success(f"Found {len(tickers)} S&P 500 stocks")
    
    # Download data for all tickers
    ticker_data = download_sp500_data(tickers)
    
    if not ticker_data:
        st.error("Failed to download S&P 500 data")
        return None
    
    st.success(f"Successfully downloaded data for {len(ticker_data)} stocks")
    
    # Analyze each ticker
    signal_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    ticker_list = list(ticker_data.items())
    
    for idx, (ticker, df) in enumerate(ticker_list):
        status_text.text(f"Analyzing {ticker}... ({idx+1}/{len(ticker_list)})")
        progress_bar.progress((idx + 1) / len(ticker_list))
        
        try:
            # Calculate indicators
            indicators_df = calculate_all_indicators(0, ticker, df)
            
            # Analyze signals
            signal_analysis = analyze_buy_sell_signals(indicators_df)
            
            # Create a dictionary with individual indicator signals
            row_data = {
                'Symbol': ticker,
                'Name': ticker,  # Could fetch company name if needed
                'Signal': signal_analysis['overall'],
                'Buy Count': signal_analysis['buy_count'],
                'Sell Count': signal_analysis['sell_count'],
                'Neutral Count': signal_analysis['neutral_count'],
                'Buy %': signal_analysis['buy_percentage'],
                'Sell %': signal_analysis['sell_percentage'],
                'Trend': signal_analysis['trend_strength'],
                'Close': indicators_df.iloc[-1]['Close']
            }
            
            # Add individual indicator signals and their detailed reasons
            for indicator, signal, reason in signal_analysis['signals']:
                row_data[indicator] = signal
                row_data[f"{indicator} Detail"] = f"{signal} - {reason}"
            
            signal_data.append(row_data)
        except Exception as e:
            # Skip tickers with errors
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(signal_data)

def analyze_sp500_macd_crossovers(lookback_days=5):
    """
    Analyze MACD crossovers for all S&P 500 stocks.
    
    Args:
        lookback_days: Number of days to look back for crossovers
        
    Returns:
        DataFrame with MACD crossover information
    """
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    
    if not tickers:
        st.error("Failed to fetch S&P 500 ticker list")
        return None
    
    st.success(f"Found {len(tickers)} S&P 500 stocks")
    
    # Download data for all tickers
    ticker_data = download_sp500_data(tickers, period="1y")  # Use 1 year for better data availability
    
    if not ticker_data or len(ticker_data) == 0:
        st.error("Failed to download S&P 500 data. Please check your internet connection and try again.")
        return None
    
    st.info(f"Analyzing {len(ticker_data)} stocks for MACD crossovers...")
    
    # Analyze each ticker for MACD crossovers
    crossover_data = []
    error_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    ticker_list = list(ticker_data.items())
    
    for idx, (ticker, df) in enumerate(ticker_list):
        status_text.text(f"Checking {ticker} for MACD crossovers... ({idx+1}/{len(ticker_list)})")
        progress_bar.progress((idx + 1) / len(ticker_list))
        
        try:
            # Calculate indicators
            indicators_df = calculate_all_indicators(0, ticker, df)
            
            # Detect MACD crossover
            crossover = detect_macd_crossover(indicators_df, lookback_days)
            
            if crossover:
                crossover_data.append({
                    'Symbol': ticker,
                    'Crossover Type': crossover['type'],
                    'Crossover Date': crossover['date'].strftime('%Y-%m-%d'),
                    'Days Ago': crossover['days_ago'],
                    'Current MACD': round(crossover['macd'], 4),
                    'Current Signal': round(crossover['signal'], 4),
                    'Position': crossover['current_position'],
                    'Close': round(indicators_df.iloc[-1]['Close'], 2)
                })
        except Exception as e:
            # Skip tickers with errors
            error_count += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    if crossover_data:
        st.success(f"Found {len(crossover_data)} stocks with MACD crossovers in the last {lookback_days} days!")
        if error_count > 0:
            st.info(f"Note: {error_count} stocks were skipped due to data issues")
        return pd.DataFrame(crossover_data)
    else:
        st.warning(f"No MACD crossovers found in the last {lookback_days} days across {len(ticker_data)} analyzed stocks.")
        if error_count > 0:
            st.info(f"Note: {error_count} stocks were skipped due to data issues")
        return pd.DataFrame()  # Return empty dataframe if no crossovers

def get_fundamental_data(symbol):
    """
    Fetch fundamental data for a given ticker symbol.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        Dictionary with fundamental metrics or None if error
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract fundamental metrics
        fundamentals = {
            'eps': info.get('trailingEps', None),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'dividend_rate': info.get('dividendRate', None),
            'market_cap': info.get('marketCap', None),
            'beta': info.get('beta', None),
            'revenue': info.get('totalRevenue', None),
            'profit_margin': info.get('profitMargins', None),
            'book_value': info.get('bookValue', None),
            'enterprise_value': info.get('enterpriseValue', None),
            'current_price': info.get('currentPrice', None),
            'company_name': info.get('longName', symbol)
        }
        
        return fundamentals
        
    except Exception as e:
        st.error(f"Error fetching fundamental data: {str(e)}")
        return None

def display_fundamental_analysis(symbol):
    """
    Display fundamental analysis section for a ticker.
    
    Args:
        symbol: Ticker symbol
    """
    st.subheader(f"📈 Fundamental Analysis for {symbol}")
    
    with st.spinner(f"Fetching fundamental data for {symbol}..."):
        fundamentals = get_fundamental_data(symbol)
    
    if not fundamentals:
        st.error("Unable to fetch fundamental data for this ticker.")
        return
    
    # Display company name
    if fundamentals.get('company_name'):
        st.markdown(f"### {fundamentals['company_name']}")
    
    st.markdown("---")
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### Earnings Per Share (EPS)")
        if fundamentals.get('eps') is not None:
            eps = fundamentals['eps']
            st.metric("Trailing EPS", f"${eps:.2f}")
            if eps > 0:
                st.success("✓ Positive earnings")
            else:
                st.warning("⚠ Negative earnings")
        else:
            st.info("N/A")
    
    with col2:
        st.markdown("#### Price to Earnings (P/E)")
        if fundamentals.get('pe_ratio') is not None:
            pe = fundamentals['pe_ratio']
            st.metric("Trailing P/E", f"{pe:.2f}")
            
            # P/E interpretation
            if pe < 15:
                st.success("✓ Potentially undervalued")
            elif pe < 25:
                st.info("○ Moderate valuation")
            else:
                st.warning("⚠ Potentially overvalued")
        else:
            st.info("N/A")
        
        if fundamentals.get('forward_pe') is not None:
            st.metric("Forward P/E", f"{fundamentals['forward_pe']:.2f}")
        
        if fundamentals.get('peg_ratio') is not None:
            peg = fundamentals['peg_ratio']
            st.metric("PEG Ratio", f"{peg:.2f}")
            if peg < 1:
                st.success("✓ Undervalued growth")
            elif peg < 2:
                st.info("○ Fair value")
            else:
                st.warning("⚠ Overvalued")
    
    with col3:
        st.markdown("#### Price to Book (P/B)")
        if fundamentals.get('pb_ratio') is not None:
            pb = fundamentals['pb_ratio']
            st.metric("P/B Ratio", f"{pb:.2f}")
            
            # P/B interpretation
            if pb < 1:
                st.success("✓ Trading below book value")
            elif pb < 3:
                st.info("○ Moderate valuation")
            else:
                st.warning("⚠ Trading at premium")
        else:
            st.info("N/A")
        
        if fundamentals.get('book_value') is not None:
            st.metric("Book Value", f"${fundamentals['book_value']:.2f}")
    
    with col4:
        st.markdown("#### Dividend Yield")
        if fundamentals.get('dividend_yield') is not None:
            div_yield = fundamentals['dividend_yield']
            st.metric("Yield", f"{div_yield:.2f}%")
            
            # Dividend yield interpretation
            if div_yield > 4:
                st.success("✓ High dividend yield")
            elif div_yield > 2:
                st.info("○ Moderate dividend yield")
            elif div_yield > 0:
                st.warning("⚠ Low dividend yield")
            else:
                st.info("No dividend")
        else:
            st.info("N/A")
        
        if fundamentals.get('dividend_rate') is not None:
            st.metric("Annual Dividend", f"${fundamentals['dividend_rate']:.2f}")
    
    # Additional metrics in expandable section
    with st.expander("📊 Additional Fundamental Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if fundamentals.get('market_cap') is not None:
                market_cap = fundamentals['market_cap']
                if market_cap >= 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                elif market_cap >= 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
            
            if fundamentals.get('beta') is not None:
                beta = fundamentals['beta']
                st.metric("Beta", f"{beta:.2f}")
                if beta > 1:
                    st.info("More volatile than market")
                elif beta < 1:
                    st.info("Less volatile than market")
        
        with col2:
            if fundamentals.get('revenue') is not None:
                revenue = fundamentals['revenue']
                if revenue >= 1e12:
                    st.metric("Total Revenue", f"${revenue/1e12:.2f}T")
                elif revenue >= 1e9:
                    st.metric("Total Revenue", f"${revenue/1e9:.2f}B")
                else:
                    st.metric("Total Revenue", f"${revenue/1e6:.2f}M")
            
            if fundamentals.get('profit_margin') is not None:
                profit_margin = fundamentals['profit_margin'] * 100
                st.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        with col3:
            if fundamentals.get('enterprise_value') is not None:
                ev = fundamentals['enterprise_value']
                if ev >= 1e12:
                    st.metric("Enterprise Value", f"${ev/1e12:.2f}T")
                elif ev >= 1e9:
                    st.metric("Enterprise Value", f"${ev/1e9:.2f}B")
                else:
                    st.metric("Enterprise Value", f"${ev/1e6:.2f}M")
            
            if fundamentals.get('current_price') is not None:
                st.metric("Current Price", f"${fundamentals['current_price']:.2f}")
    
    st.markdown("---")
    
    # Valuation summary
    st.markdown("#### 📋 Valuation Summary")
    
    valuation_score = 0
    valuation_notes = []
    
    if fundamentals.get('eps') is not None and fundamentals['eps'] > 0:
        valuation_score += 1
        valuation_notes.append("✓ Positive earnings")
    
    if fundamentals.get('pe_ratio') is not None and fundamentals['pe_ratio'] < 25:
        valuation_score += 1
        valuation_notes.append("✓ Reasonable P/E ratio")
    
    if fundamentals.get('pb_ratio') is not None and fundamentals['pb_ratio'] < 3:
        valuation_score += 1
        valuation_notes.append("✓ Reasonable P/B ratio")
    
    if fundamentals.get('dividend_yield') is not None and fundamentals['dividend_yield'] > 2:
        valuation_score += 1
        valuation_notes.append("✓ Pays dividend (>2%)")
    
    # Display valuation score
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Valuation Score", f"{valuation_score}/4")
    
    with col2:
        for note in valuation_notes:
            st.markdown(note)
    
    st.info("**Note:** Fundamental analysis should be combined with technical analysis and thorough research before making investment decisions.")

def main():
    """
    Main Streamlit app function.
    """
    st.title("📊 ETF Technical Indicators Analysis")
    
    # Add view mode selector
    view_mode = st.radio("Select View", 
                        ["Individual ETF Analysis", "All ETFs Signal Overview", "Custom Ticker Analysis", "S&P 500 Analysis"], 
                        horizontal=True)
    
    st.markdown("---")
    
    # Initialize indicators database
    indicators_conn = sqlite3.connect(INDICATORS_DATABASE_PATH)
    create_indicators_table(indicators_conn)
    indicators_conn.close()
    
    # Connect to source database
    source_conn = sqlite3.connect(SOURCE_DATABASE_PATH)
    
    try:
        # Get list of ETFs
        etf_list = get_etf_list(source_conn)
        
        if not etf_list:
            st.error("No ETFs found in the database.")
            return
        
        if view_mode == "Custom Ticker Analysis":
            # Show custom ticker analysis
            st.header("🔍 Custom Ticker Analysis")
            st.markdown("Enter any ticker symbol to download data from Yahoo Finance and analyze technical indicators.")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                ticker_input = st.text_input(
                    "Enter Ticker Symbol",
                    placeholder="e.g., AAPL, SPY, MSFT",
                    help="Enter any valid ticker symbol from Yahoo Finance"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_button = st.button("📊 Analyze", type="primary")
            
            if analyze_button and ticker_input:
                ticker_symbol = ticker_input.strip().upper()
                
                # Analyze the ticker
                df = analyze_custom_ticker(ticker_symbol)
                
                if df is not None:
                    # Store in session state
                    st.session_state['custom_ticker'] = ticker_symbol
                    st.session_state['custom_df'] = df
                    st.success(f"Successfully loaded {len(df)} data points for {ticker_symbol}")
            
            # Display analysis if data is available
            if 'custom_ticker' in st.session_state and 'custom_df' in st.session_state:
                ticker_symbol = st.session_state['custom_ticker']
                df = st.session_state['custom_df']
                
                st.markdown("---")
                
                # Time Frame Selection
                st.sidebar.header("Time Frame")
                
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                
                # Quick time frame buttons
                time_frame = st.sidebar.radio(
                    "Select Time Frame",
                    ["All Time", "YTD", "1 Year", "3 Years", "5 Years", "Custom"],
                    index=0,
                    key="custom_timeframe"
                )
                
                if time_frame == "YTD":
                    start_date = datetime(max_date.year, 1, 1).date()
                    end_date = max_date
                elif time_frame == "1 Year":
                    start_date = (datetime.now() - timedelta(days=365)).date()
                    end_date = max_date
                elif time_frame == "3 Years":
                    start_date = (datetime.now() - timedelta(days=365*3)).date()
                    end_date = max_date
                elif time_frame == "5 Years":
                    start_date = (datetime.now() - timedelta(days=365*5)).date()
                    end_date = max_date
                elif time_frame == "Custom":
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="custom_start")
                    with col2:
                        end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="custom_end")
                else:  # All Time
                    start_date = min_date
                    end_date = max_date
                
                # Filter dataframe by date range
                df_filtered = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()
                
                # Analyze and display buy/sell signals
                signal_analysis = analyze_buy_sell_signals(df_filtered)
                display_buy_sell_signal(signal_analysis, ticker_symbol)
                
                st.markdown("---")
                
                # Display indicator values
                display_indicator_values(df_filtered, ticker_symbol)
                
                st.markdown("---")
                
                # Display fundamental analysis
                display_fundamental_analysis(ticker_symbol)
                
                st.markdown("---")
                
                # Chart selection
                st.sidebar.markdown("---")
                st.sidebar.header("Chart Selection")
                chart_type = st.sidebar.selectbox(
                    "Select Indicator Chart",
                    ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Stochastic", "OBV", "ADX"],
                    key="custom_chart_type"
                )
                
                # Moving Average Selection
                selected_ma = None
                if chart_type == "Moving Averages":
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Moving Averages")
                    
                    ma_options = {
                        'SMA_20': 'SMA 20',
                        'SMA_50': 'SMA 50',
                        'SMA_100': 'SMA 100',
                        'SMA_200': 'SMA 200',
                        'EMA_12': 'EMA 12',
                        'EMA_26': 'EMA 26',
                        'EMA_50': 'EMA 50'
                    }
                    
                    col1, col2 = st.sidebar.columns(2)
                    select_all = col1.button("Select All", key="custom_select_all")
                    deselect_all = col2.button("Deselect All", key="custom_deselect_all")
                    
                    selected_ma = []
                    for ma_key, ma_label in ma_options.items():
                        if ma_key in df_filtered.columns:
                            default_value = True
                            if deselect_all:
                                default_value = False
                            elif select_all:
                                default_value = True
                            
                            if st.sidebar.checkbox(ma_label, value=default_value, key=f"custom_cb_{ma_key}"):
                                selected_ma.append(ma_key)
                
                # Display chart
                fig = plot_price_with_indicators(df_filtered, ticker_symbol, chart_type, selected_ma)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.markdown("---")
                with st.expander("📋 View Full Data Table"):
                    display_cols = ['Date', 'Close', 'RSI_14', 'MACD', 'MACD_Signal', 
                                  'SMA_20', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
                    display_cols = [col for col in display_cols if col in df_filtered.columns]
                    
                    st.dataframe(
                        df_filtered[display_cols].sort_values('Date', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                
                # Download button
                st.markdown("---")
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {ticker_symbol} Indicators as CSV",
                    data=csv,
                    file_name=f"{ticker_symbol}_technical_indicators_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )
                
                # Show info
                st.sidebar.markdown("---")
                st.sidebar.info(f"""
                **Ticker:** {ticker_symbol}  
                **Data Points:** {len(df_filtered)}  
                **Date Range:** {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}  
                **Indicators:** {len([col for col in df_filtered.columns if col not in ['Date', 'Close', 'etf_id', 'symbol', 'id']])}
                """)
            
            else:
                st.info("👆 Enter a ticker symbol above and click 'Analyze' to get started.")
            
            return
        
        if view_mode == "S&P 500 Analysis":
            # Show S&P 500 analysis
            st.header("📈 S&P 500 Buy/Sell Signals")
            st.markdown("This view analyzes technical indicators across all S&P 500 stocks to identify buy and sell opportunities.")
            
            # Add tabs for different analysis types
            analysis_tab = st.radio(
                "Select Analysis Type",
                ["Full Technical Analysis", "MACD Crossovers (Last 5 Days)"],
                horizontal=True
            )
            
            st.markdown("---")
            
            if analysis_tab == "MACD Crossovers (Last 5 Days)":
                # MACD Crossover Analysis
                st.subheader("🔄 Recent MACD Crossovers")
                st.markdown("Identify stocks with MACD crossovers in the past 5 days - potential entry/exit points.")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    lookback_days = st.number_input("Lookback Days", min_value=1, max_value=30, value=5, step=1)
                
                if st.button("🔍 Find MACD Crossovers", type="primary"):
                    with st.spinner("Analyzing MACD crossovers..."):
                        crossover_df = analyze_sp500_macd_crossovers(lookback_days)
                        
                        if crossover_df is not None and not crossover_df.empty:
                            st.session_state['sp500_crossovers_df'] = crossover_df
                            st.success(f"Found {len(crossover_df)} stocks with MACD crossovers!")
                        else:
                            st.warning("No MACD crossovers found in the specified timeframe.")
                
                # Display crossover results
                if 'sp500_crossovers_df' in st.session_state:
                    crossover_df = st.session_state['sp500_crossovers_df']
                    
                    if not crossover_df.empty:
                        st.markdown("---")
                        
                        # Add filter options for crossovers
                        st.subheader("🔍 Filter MACD Crossovers")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            filter_crossover_type = st.multiselect(
                                "Filter by Crossover Type",
                                options=['BUY', 'SELL'],
                                default=['BUY', 'SELL'],
                                key='crossover_type_filter'
                            )
                        
                        with col2:
                            max_days_ago = st.slider(
                                "Maximum Days Ago",
                                min_value=1,
                                max_value=int(crossover_df['Days Ago'].max()) if len(crossover_df) > 0 else 30,
                                value=int(crossover_df['Days Ago'].max()) if len(crossover_df) > 0 else 30,
                                step=1,
                                key='days_ago_filter'
                            )
                        
                        with col3:
                            min_price = st.number_input(
                                "Minimum Price ($)",
                                min_value=0.0,
                                value=0.0,
                                step=10.0,
                                key='min_price_filter'
                            )
                        
                        # Apply filters
                        filtered_crossover_df = crossover_df[
                            (crossover_df['Crossover Type'].isin(filter_crossover_type)) &
                            (crossover_df['Days Ago'] <= max_days_ago) &
                            (crossover_df['Close'] >= min_price)
                        ].copy()
                        
                        st.markdown("---")
                        
                        # Summary metrics for filtered data
                        col1, col2, col3 = st.columns(3)
                        buy_crossovers = len(filtered_crossover_df[filtered_crossover_df['Crossover Type'] == 'BUY'])
                        sell_crossovers = len(filtered_crossover_df[filtered_crossover_df['Crossover Type'] == 'SELL'])
                        
                        with col1:
                            st.metric("🟢 Bullish Crossovers", buy_crossovers, help="MACD crossed above Signal line")
                        with col2:
                            st.metric("🔴 Bearish Crossovers", sell_crossovers, help="MACD crossed below Signal line")
                        with col3:
                            st.metric("📊 Total Crossovers", len(filtered_crossover_df))
                        
                        if filtered_crossover_df.empty:
                            st.warning("No MACD crossovers match the current filter criteria.")
                        else:
                            st.markdown("---")
                        
                            # Separate into Buy and Sell (using filtered data)
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("🟢 Bullish MACD Crossovers (BUY)")
                                buy_df = filtered_crossover_df[filtered_crossover_df['Crossover Type'] == 'BUY'].copy()
                                if not buy_df.empty:
                                    buy_df = buy_df.sort_values('Days Ago')
                                    
                                    # Style the dataframe
                                    def highlight_buy(row):
                                        return ['background-color: #00CC0033'] * len(row)
                                    
                                    styled_buy = buy_df.style.apply(highlight_buy, axis=1)
                                    st.dataframe(styled_buy, use_container_width=True, height=400)
                                    
                                    # Download button
                                    csv_buy = buy_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Bullish Crossovers",
                                        data=csv_buy,
                                        file_name=f"bullish_macd_crossovers_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv",
                                        key="download_buy_crossovers"
                                    )
                                else:
                                    st.info("No bullish MACD crossovers found.")
                            
                            with col2:
                                st.subheader("🔴 Bearish MACD Crossovers (SELL)")
                                sell_df = filtered_crossover_df[filtered_crossover_df['Crossover Type'] == 'SELL'].copy()
                                if not sell_df.empty:
                                    sell_df = sell_df.sort_values('Days Ago')
                                
                                    # Style the dataframe
                                    def highlight_sell(row):
                                        return ['background-color: #CC000033'] * len(row)
                                    
                                    styled_sell = sell_df.style.apply(highlight_sell, axis=1)
                                    st.dataframe(styled_sell, use_container_width=True, height=400)
                                    
                                    # Download button
                                    csv_sell = sell_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Bearish Crossovers",
                                        data=csv_sell,
                                        file_name=f"bearish_macd_crossovers_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv",
                                        key="download_sell_crossovers"
                                    )
                                else:
                                    st.info("No bearish MACD crossovers found.")
                            
                            # Full list (filtered)
                            st.markdown("---")
                            st.subheader("📋 All MACD Crossovers (Filtered)")
                            
                            # Style based on crossover type
                            def highlight_crossover(row):
                                if row['Crossover Type'] == 'BUY':
                                    return ['background-color: #00CC0033'] * len(row)
                                else:
                                    return ['background-color: #CC000033'] * len(row)
                            
                            styled_all = filtered_crossover_df.style.apply(highlight_crossover, axis=1)
                            st.dataframe(styled_all, use_container_width=True)
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_filtered = filtered_crossover_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Filtered Crossovers",
                                    data=csv_filtered,
                                    file_name=f"filtered_macd_crossovers_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    key="download_filtered_crossovers"
                                )
                            with col2:
                                csv_all = crossover_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download All Crossovers",
                                    data=csv_all,
                                    file_name=f"all_macd_crossovers_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    key="download_all_crossovers"
                                )
                    else:
                        st.info("No MACD crossovers found in the specified timeframe.")
                
                return
            
            # Full Technical Analysis (original code)
            st.warning("⚠️ This analysis will download data for 500+ stocks and may take several minutes.")
            
            if st.button("🚀 Analyze S&P 500", type="primary"):
                with st.spinner("Fetching and analyzing S&P 500 data..."):
                    signals_df = analyze_sp500_signals()
                    
                    if signals_df is not None and not signals_df.empty:
                        st.session_state['sp500_signals_df'] = signals_df
                        st.success(f"Analysis complete! Analyzed {len(signals_df)} stocks.")
                    else:
                        st.error("Failed to analyze S&P 500 stocks.")
            
            # Display results if available
            if 'sp500_signals_df' in st.session_state:
                signals_df = st.session_state['sp500_signals_df']
                
                # Add filter options
                st.markdown("---")
                st.subheader("🔍 Filter Results")
                
                # Check which indicator columns are available
                available_indicators = [col for col in signals_df.columns if col not in 
                                       ['Symbol', 'Name', 'Signal', 'Buy Count', 'Sell Count', 
                                        'Neutral Count', 'Buy %', 'Sell %', 'Trend', 'Close']
                                       and not col.endswith(' Detail')]
                available_detail_indicators = [col for col in signals_df.columns if col.endswith(' Detail')]
                
                with st.expander("⚙️ Basic Filters", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        filter_signal = st.multiselect(
                            "Filter by Overall Signal",
                            options=['BUY', 'SELL', 'NEUTRAL'],
                            default=['BUY', 'SELL', 'NEUTRAL'],
                            key='sp500_signal_filter'
                        )
                    
                    with col2:
                        min_buy_pct = st.slider(
                            "Minimum Buy %",
                            min_value=0,
                            max_value=100,
                            value=0,
                            step=10,
                            key='sp500_buy_pct_filter'
                        )
                    
                    with col3:
                        trend_filter = st.multiselect(
                            "Filter by Trend Strength",
                            options=['Strong', 'Moderate', 'Weak', 'N/A'],
                            default=['Strong', 'Moderate', 'Weak', 'N/A'],
                            key='sp500_trend_filter'
                        )
                
                # Advanced indicator-specific filters
                if available_indicators or available_detail_indicators:
                    with st.expander("🎯 Advanced Filters - Filter by Specific Indicators"):
                        st.markdown("**Select specific indicator signals to filter by. Only stocks matching ALL selected criteria will be shown.**")
                        st.markdown("*Example: Select RSI (14) = BUY and MACD = BUY to find stocks with both signals*")
                        
                        # Tabs for Signal Type vs Detailed Signal
                        tab1, tab2 = st.tabs(["Filter by Signal Type (BUY/SELL/NEUTRAL)", "Filter by Detailed Signal"])
                        
                        indicator_filters = {}
                        
                        with tab1:
                            st.markdown("*Filter by the general signal classification*")
                            # Create columns for better layout
                            num_cols = 3
                            cols = st.columns(num_cols)
                            
                            for idx, indicator in enumerate(available_indicators):
                                with cols[idx % num_cols]:
                                    # Get unique values for this indicator in the dataset
                                    unique_values = signals_df[indicator].dropna().unique()
                                    if len(unique_values) > 0:
                                        selected_values = st.multiselect(
                                            f"{indicator}",
                                            options=sorted(unique_values),
                                            default=[],
                                            key=f'sp500_filter_{indicator}'
                                        )
                                        if selected_values:
                                            indicator_filters[indicator] = selected_values
                        
                        with tab2:
                            st.markdown("*Filter by the detailed signal description (e.g., 'Oversold at 36.25', 'Bearish crossover')*")
                            # Create columns for better layout
                            num_cols = 2
                            cols = st.columns(num_cols)
                            
                            for idx, indicator in enumerate(available_detail_indicators):
                                with cols[idx % num_cols]:
                                    # Get unique values for this indicator in the dataset
                                    unique_values = signals_df[indicator].dropna().unique()
                                    if len(unique_values) > 0:
                                        selected_values = st.multiselect(
                                            f"{indicator.replace(' Detail', '')}",
                                            options=sorted(unique_values),
                                            default=[],
                                            key=f'sp500_filter_detail_{indicator}',
                                            help=f"Filter by specific {indicator.replace(' Detail', '')} conditions"
                                        )
                                        if selected_values:
                                            indicator_filters[indicator] = selected_values
                
                # Apply filters
                filtered_df = signals_df[
                    (signals_df['Signal'].isin(filter_signal)) &
                    (signals_df['Buy %'] >= min_buy_pct) &
                    (signals_df['Trend'].isin(trend_filter))
                ].copy()
                
                # Apply indicator-specific filters
                if available_indicators:
                    for indicator, values in indicator_filters.items():
                        if values:
                            filtered_df = filtered_df[filtered_df[indicator].isin(values)]
                
                st.markdown("---")
                
                if not filtered_df.empty:
                    # Display the chart
                    st.subheader(f"Filtered Results: {len(filtered_df)} stocks")
                    plot_all_signals_chart(filtered_df)
                    
                    # Top BUY recommendations
                    st.markdown("---")
                    buy_stocks = filtered_df[filtered_df['Signal'] == 'BUY'].sort_values('Buy %', ascending=False)
                    
                    if not buy_stocks.empty:
                        st.subheader("🟢 Top BUY Recommendations")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Strong BUY Signals", len(buy_stocks[buy_stocks['Buy %'] >= 70]))
                        with col2:
                            st.metric("Moderate BUY Signals", len(buy_stocks[(buy_stocks['Buy %'] >= 60) & (buy_stocks['Buy %'] < 70)]))
                        
                        # Show top 10 buy recommendations
                        st.dataframe(
                            buy_stocks.head(10)[['Symbol', 'Signal', 'Buy %', 'Sell %', 'Trend', 'Close']],
                            use_container_width=True
                        )
                    
                    # Top SELL recommendations
                    sell_stocks = filtered_df[filtered_df['Signal'] == 'SELL'].sort_values('Sell %', ascending=False)
                    
                    if not sell_stocks.empty:
                        st.markdown("---")
                        st.subheader("🔴 Top SELL Recommendations")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Strong SELL Signals", len(sell_stocks[sell_stocks['Sell %'] >= 70]))
                        with col2:
                            st.metric("Moderate SELL Signals", len(sell_stocks[(sell_stocks['Sell %'] >= 60) & (sell_stocks['Sell %'] < 70)]))
                        
                        # Show top 10 sell recommendations
                        st.dataframe(
                            sell_stocks.head(10)[['Symbol', 'Signal', 'Buy %', 'Sell %', 'Trend', 'Close']],
                            use_container_width=True
                        )
                    
                    # Download button
                    st.markdown("---")
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Results as CSV",
                        data=csv,
                        file_name=f"sp500_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Full results download
                    csv_all = signals_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All S&P 500 Results as CSV",
                        data=csv_all,
                        file_name=f"sp500_all_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Detailed Stock Analysis Section
                    st.markdown("---")
                    st.header("🔍 Detailed Stock Analysis")
                    st.markdown("Select a stock from the analyzed list to view detailed technical indicators and charts.")
                    
                    # Create selectbox with sorted ticker list
                    ticker_options = sorted(signals_df['Symbol'].tolist())
                    selected_ticker = st.selectbox(
                        "Select Stock to Analyze",
                        options=ticker_options,
                        key="sp500_ticker_select"
                    )
                    
                    if selected_ticker:
                        # Check if we have the data cached
                        cache_key = f'sp500_detailed_{selected_ticker}'
                        
                        if cache_key not in st.session_state:
                            with st.spinner(f"Loading detailed data for {selected_ticker}..."):
                                # Download and analyze the ticker
                                df, error = download_ticker_data(selected_ticker, period="2y")
                                if df is not None and len(df) >= 200:
                                    indicators_df = calculate_all_indicators(0, selected_ticker, df)
                                    st.session_state[cache_key] = indicators_df
                                else:
                                    st.error(f"Unable to load detailed data for {selected_ticker}")
                        
                        if cache_key in st.session_state:
                            df = st.session_state[cache_key]
                            
                            # Time Frame Selection
                            st.sidebar.markdown("---")
                            st.sidebar.header("📅 Detailed View Settings")
                            
                            min_date = df['Date'].min().date()
                            max_date = df['Date'].max().date()
                            
                            time_frame = st.sidebar.radio(
                                "Select Time Frame",
                                ["All Time", "YTD", "1 Year", "3 Years", "6 Months", "3 Months", "Custom"],
                                index=2,  # Default to 1 Year
                                key="sp500_timeframe"
                            )
                            
                            if time_frame == "YTD":
                                start_date = datetime(max_date.year, 1, 1).date()
                                end_date = max_date
                            elif time_frame == "1 Year":
                                start_date = (datetime.now() - timedelta(days=365)).date()
                                end_date = max_date
                            elif time_frame == "3 Years":
                                start_date = (datetime.now() - timedelta(days=365*3)).date()
                                end_date = max_date
                            elif time_frame == "6 Months":
                                start_date = (datetime.now() - timedelta(days=180)).date()
                                end_date = max_date
                            elif time_frame == "3 Months":
                                start_date = (datetime.now() - timedelta(days=90)).date()
                                end_date = max_date
                            elif time_frame == "Custom":
                                col1, col2 = st.sidebar.columns(2)
                                with col1:
                                    start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="sp500_start")
                                with col2:
                                    end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="sp500_end")
                            else:  # All Time
                                start_date = min_date
                                end_date = max_date
                            
                            # Filter dataframe by date range
                            df_filtered = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()
                            
                            # Analyze and display buy/sell signals
                            signal_analysis = analyze_buy_sell_signals(df_filtered)
                            display_buy_sell_signal(signal_analysis, selected_ticker)
                            
                            st.markdown("---")
                            
                            # Display indicator values
                            display_indicator_values(df_filtered, selected_ticker)
                            
                            st.markdown("---")
                            
                            # Display fundamental analysis
                            display_fundamental_analysis(selected_ticker)
                            
                            st.markdown("---")
                            
                            # Chart selection with indicator toggles
                            st.sidebar.markdown("---")
                            st.sidebar.header("📊 Chart Options")
                            
                            chart_type = st.sidebar.selectbox(
                                "Select Indicator Chart",
                                ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Stochastic", "OBV"],
                                key="sp500_chart_type"
                            )
                            
                            # Moving Average Selection
                            selected_ma = None
                            if chart_type == "Moving Averages":
                                st.sidebar.markdown("---")
                                st.sidebar.subheader("Toggle Indicators")
                                
                                ma_options = {
                                    'SMA_20': 'SMA 20',
                                    'SMA_50': 'SMA 50',
                                    'SMA_100': 'SMA 100',
                                    'SMA_200': 'SMA 200',
                                    'EMA_12': 'EMA 12',
                                    'EMA_26': 'EMA 26',
                                    'EMA_50': 'EMA 50'
                                }
                                
                                col1, col2 = st.sidebar.columns(2)
                                select_all = col1.button("Select All", key="sp500_select_all")
                                deselect_all = col2.button("Deselect All", key="sp500_deselect_all")
                                
                                selected_ma = []
                                for ma_key, ma_label in ma_options.items():
                                    if ma_key in df_filtered.columns:
                                        default_value = True
                                        if deselect_all:
                                            default_value = False
                                        elif select_all:
                                            default_value = True
                                        
                                        if st.sidebar.checkbox(ma_label, value=default_value, key=f"sp500_cb_{ma_key}"):
                                            selected_ma.append(ma_key)
                            
                            # Display chart
                            fig = plot_price_with_indicators(df_filtered, selected_ticker, chart_type, selected_ma)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table
                            st.markdown("---")
                            with st.expander("📋 View Full Data Table"):
                                display_cols = ['Date', 'Close', 'RSI_14', 'MACD', 'MACD_Signal', 
                                              'SMA_20', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
                                display_cols = [col for col in display_cols if col in df_filtered.columns]
                                
                                st.dataframe(
                                    df_filtered[display_cols].sort_values('Date', ascending=False),
                                    use_container_width=True,
                                    height=400
                                )
                            
                            # Download button
                            st.markdown("---")
                            csv = df_filtered.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {selected_ticker} Indicators as CSV",
                                data=csv,
                                file_name=f"{selected_ticker}_technical_indicators_{start_date}_to_{end_date}.csv",
                                mime="text/csv",
                                key="sp500_download_detailed"
                            )
                
                else:
                    st.warning("No stocks match the current filter criteria.")
            else:
                st.info("👆 Click the 'Analyze S&P 500' button above to get started.")
            
            return
        
        if view_mode == "All ETFs Signal Overview":
            # Show all ETFs signal overview
            st.header("📊 All ETFs Buy/Sell Signals")
            st.markdown("This view analyzes technical indicators across all ETFs to identify buy and sell opportunities.")
            
            if st.button("🔄 Refresh All Signals", type="primary"):
                with st.spinner("Analyzing all ETFs..."):
                    signals_df = get_all_etf_signals(etf_list)
                    st.session_state['signals_df'] = signals_df
            
            # Load or use cached signals
            if 'signals_df' not in st.session_state:
                with st.spinner("Analyzing all ETFs..."):
                    signals_df = get_all_etf_signals(etf_list)
                    st.session_state['signals_df'] = signals_df
            else:
                signals_df = st.session_state['signals_df']
            
            # Add filter options
            st.markdown("---")
            st.subheader("🔍 Filter Results")
            
            # Check which indicator columns are available
            available_indicators = [col for col in signals_df.columns if col not in 
                                   ['Symbol', 'Name', 'Signal', 'Buy Count', 'Sell Count', 
                                    'Neutral Count', 'Buy %', 'Sell %', 'Trend', 'Close'] 
                                   and not col.endswith(' Detail')]
            available_detail_indicators = [col for col in signals_df.columns if col.endswith(' Detail')]
            
            with st.expander("⚙️ Basic Filters", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    filter_signal = st.multiselect(
                        "Filter by Overall Signal",
                        options=['BUY', 'SELL', 'NEUTRAL'],
                        default=['BUY', 'SELL', 'NEUTRAL'],
                        key='etf_signal_filter'
                    )
                
                with col2:
                    min_buy_pct = st.slider(
                        "Minimum Buy %",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=10,
                        key='etf_buy_pct_filter'
                    )
            
            # Advanced indicator-specific filters
            if available_indicators or available_detail_indicators:
                with st.expander("🎯 Advanced Filters - Filter by Specific Indicators"):
                    st.markdown("**Select specific indicator signals to filter by. Only stocks matching ALL selected criteria will be shown.**")
                    
                    # Tabs for Signal Type vs Detailed Signal
                    tab1, tab2 = st.tabs(["Filter by Signal Type (BUY/SELL/NEUTRAL)", "Filter by Detailed Signal"])
                    
                    indicator_filters = {}
                    
                    with tab1:
                        st.markdown("*Filter by the general signal classification*")
                        # Create columns for better layout
                        num_cols = 3
                        cols = st.columns(num_cols)
                        
                        for idx, indicator in enumerate(available_indicators):
                            with cols[idx % num_cols]:
                                # Get unique values for this indicator in the dataset
                                unique_values = signals_df[indicator].dropna().unique()
                                if len(unique_values) > 0:
                                    selected_values = st.multiselect(
                                        f"{indicator}",
                                        options=sorted(unique_values),
                                        default=[],
                                        key=f'etf_filter_{indicator}'
                                    )
                                    if selected_values:
                                        indicator_filters[indicator] = selected_values
                    
                    with tab2:
                        st.markdown("*Filter by the detailed signal description (e.g., 'Oversold at 36.25', 'Bearish crossover')*")
                        # Create columns for better layout
                        num_cols = 2
                        cols = st.columns(num_cols)
                        
                        for idx, indicator in enumerate(available_detail_indicators):
                            with cols[idx % num_cols]:
                                # Get unique values for this indicator in the dataset
                                unique_values = signals_df[indicator].dropna().unique()
                                if len(unique_values) > 0:
                                    selected_values = st.multiselect(
                                        f"{indicator.replace(' Detail', '')}",
                                        options=sorted(unique_values),
                                        default=[],
                                        key=f'etf_filter_detail_{indicator}',
                                        help=f"Filter by specific {indicator.replace(' Detail', '')} conditions"
                                    )
                                    if selected_values:
                                        indicator_filters[indicator] = selected_values
            
            # Apply filters
            filtered_df = signals_df[
                (signals_df['Signal'].isin(filter_signal)) &
                (signals_df['Buy %'] >= min_buy_pct)
            ].copy()
            
            # Apply indicator-specific filters
            if available_indicators:
                for indicator, values in indicator_filters.items():
                    if values:
                        filtered_df = filtered_df[filtered_df[indicator].isin(values)]
            
            st.markdown("---")
            
            if not filtered_df.empty:
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total ETFs", len(filtered_df))
                with col2:
                    st.metric("BUY Signals", len(filtered_df[filtered_df['Signal'] == 'BUY']))
                with col3:
                    st.metric("SELL Signals", len(filtered_df[filtered_df['Signal'] == 'SELL']))
                with col4:
                    st.metric("NEUTRAL", len(filtered_df[filtered_df['Signal'] == 'NEUTRAL']))
                
                # Display the chart
                plot_all_signals_chart(filtered_df)
                
                # Show filtered results in tables
                st.markdown("---")
                
                # Top BUY recommendations
                buy_etfs = filtered_df[filtered_df['Signal'] == 'BUY'].sort_values('Buy %', ascending=False)
                if not buy_etfs.empty:
                    st.subheader("🟢 Top BUY Recommendations")
                    st.dataframe(
                        buy_etfs[['Symbol', 'Signal', 'Buy %', 'Sell %', 'Close']],
                        use_container_width=True
                    )
                
                # Top SELL recommendations
                sell_etfs = filtered_df[filtered_df['Signal'] == 'SELL'].sort_values('Sell %', ascending=False)
                if not sell_etfs.empty:
                    st.markdown("---")
                    st.subheader("🔴 Top SELL Recommendations")
                    st.dataframe(
                        sell_etfs[['Symbol', 'Signal', 'Buy %', 'Sell %', 'Close']],
                        use_container_width=True
                    )
                
                # NEUTRAL signals
                neutral_etfs = filtered_df[filtered_df['Signal'] == 'NEUTRAL'].sort_values('Buy %', ascending=False)
                if not neutral_etfs.empty:
                    st.markdown("---")
                    st.subheader("🟠 NEUTRAL Signals")
                    st.dataframe(
                        neutral_etfs[['Symbol', 'Signal', 'Buy %', 'Sell %', 'Close']],
                        use_container_width=True
                    )
                
                # Download buttons
                st.markdown("---")
                csv_filtered = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Results as CSV",
                    data=csv_filtered,
                    file_name=f"etf_signals_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key='download_filtered'
                )
                
                csv_all = signals_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All ETF Signals as CSV",
                    data=csv_all,
                    file_name=f"all_etf_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key='download_all'
                )
            else:
                st.warning("No ETFs match the current filter criteria.")
            
            return
        
        # Create ETF selection dropdown
        etf_dict = {f"{symbol} - {name}": (etf_id, symbol) for etf_id, symbol, name in etf_list}
        
        # Sidebar controls
        st.sidebar.header("ETF Selection")
        selected_etf_display = st.sidebar.selectbox("Select ETF", list(etf_dict.keys()))
        etf_id, symbol = etf_dict[selected_etf_display]
        
        st.sidebar.markdown("---")
        st.sidebar.header("Options")
        force_recalc = st.sidebar.checkbox("Force Recalculate", value=False,
                                          help="Recalculate indicators even if they exist")
        
        # Calculate/Get indicators
        with st.spinner(f"Loading indicators for {symbol}..."):
            df = get_or_create_indicators(etf_id, symbol, force_recalculate=force_recalc)
        
        if df is None:
            st.error(f"Insufficient data for {symbol}. Need at least 200 data points.")
            return
        
        # Time Frame Selection
        st.sidebar.markdown("---")
        st.sidebar.header("Time Frame")
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        # Quick time frame buttons
        time_frame = st.sidebar.radio(
            "Select Time Frame",
            ["All Time", "YTD", "1 Year", "3 Years", "5 Years", "Custom"],
            index=0
        )
        
        if time_frame == "YTD":
            start_date = datetime(max_date.year, 1, 1).date()
            end_date = max_date
        elif time_frame == "1 Year":
            start_date = (datetime.now() - timedelta(days=365)).date()
            end_date = max_date
        elif time_frame == "3 Years":
            start_date = (datetime.now() - timedelta(days=365*3)).date()
            end_date = max_date
        elif time_frame == "5 Years":
            start_date = (datetime.now() - timedelta(days=365*5)).date()
            end_date = max_date
        elif time_frame == "Custom":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
        else:  # All Time
            start_date = min_date
            end_date = max_date
        
        # Filter dataframe by date range
        df_filtered = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)].copy()
        
        # Analyze and display buy/sell signals
        signal_analysis = analyze_buy_sell_signals(df_filtered)
        display_buy_sell_signal(signal_analysis, symbol)
        
        st.markdown("---")
        
        # Display indicator values (using filtered data)
        display_indicator_values(df_filtered, symbol)
        
        st.markdown("---")
        
        # Display fundamental analysis
        display_fundamental_analysis(symbol)
        
        st.markdown("---")
        
        # Chart selection
        st.sidebar.markdown("---")
        st.sidebar.header("Chart Selection")
        chart_type = st.sidebar.selectbox(
            "Select Indicator Chart",
            ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Stochastic", "OBV", "ADX"]
        )
        
        # Moving Average Selection
        selected_ma = None
        if chart_type == "Moving Averages":
            st.sidebar.markdown("---")
            st.sidebar.subheader("Moving Averages")
            
            ma_options = {
                'SMA_20': 'SMA 20',
                'SMA_50': 'SMA 50',
                'SMA_100': 'SMA 100',
                'SMA_200': 'SMA 200',
                'EMA_12': 'EMA 12',
                'EMA_26': 'EMA 26',
                'EMA_50': 'EMA 50'
            }
            
            # Select All / Deselect All buttons
            col1, col2 = st.sidebar.columns(2)
            select_all = col1.button("Select All")
            deselect_all = col2.button("Deselect All")
            
            selected_ma = []
            for ma_key, ma_label in ma_options.items():
                if ma_key in df_filtered.columns:
                    default_value = True
                    if deselect_all:
                        default_value = False
                    elif select_all:
                        default_value = True
                    
                    if st.sidebar.checkbox(ma_label, value=default_value, key=f"cb_{ma_key}_{time_frame}"):
                        selected_ma.append(ma_key)
        
        # Display selected chart
        fig = plot_price_with_indicators(df_filtered, symbol, chart_type, selected_ma)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.markdown("---")
        with st.expander("📋 View Full Data Table"):
            # Select columns to display
            display_cols = ['Date', 'Close', 'RSI_14', 'MACD', 'MACD_Signal', 
                          'SMA_20', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
            display_cols = [col for col in display_cols if col in df_filtered.columns]
            
            st.dataframe(
                df_filtered[display_cols].sort_values('Date', ascending=False),
                use_container_width=True,
                height=400
            )
        
        # Download button
        st.markdown("---")
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {symbol} Indicators as CSV",
            data=csv,
            file_name=f"{symbol}_technical_indicators_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )
        
        # Show info
        st.sidebar.markdown("---")
        st.sidebar.info(f"""
        **Data Points:** {len(df_filtered)}  
        **Date Range:** {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}  
        **Indicators:** {len([col for col in df_filtered.columns if col not in ['Date', 'Close', 'etf_id', 'symbol', 'id']])}
        """)
        
    finally:
        source_conn.close()

if __name__ == "__main__":
    main()

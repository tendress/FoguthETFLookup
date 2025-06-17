import sqlite3
import pandas as pd

def calculate_weighted_dividend_yields(database_path):
    """
    Calculates the weighted dividend yield for each security set and each model.
    Returns two DataFrames: one for security sets, one for models.
    """
    conn = sqlite3.connect(database_path)

    # Get all relevant data in one query
    query = """
        SELECT
            m.name AS model_name,
            ss.id AS security_set_id,
            ss.name AS security_set_name,
            etfs.symbol AS etf_symbol,
            sse.weight AS etf_weight,
            ms.weight AS security_set_weight,
            ei.dividendYield
        FROM models m
        JOIN model_security_set ms ON m.id = ms.model_id
        JOIN security_sets ss ON ms.security_set_id = ss.id
        JOIN security_sets_etfs sse ON ss.id = sse.security_set_id
        JOIN etfs ON sse.etf_id = etfs.id
        JOIN etf_infos ei ON etfs.symbol = ei.symbol
        WHERE sse.endDate IS NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate weighted dividend yield for each model
    # First, calculate the security set weighted yield, then weight by security set weight
    security_set_yields = security_set_yields.merge(
        df[['model_name', 'security_set_id', 'security_set_weight']].drop_duplicates(),
        on=['model_name', 'security_set_id'],
        how='left'
    )
    security_set_yields['model_weighted_dividend'] = (
        security_set_yields['weighted_dividendYield'] * security_set_yields['security_set_weight']
    )

    model_yields = (
        security_set_yields.groupby('model_name')
        .agg(weighted_dividendYield=('model_weighted_dividend', 'sum'))
        .reset_index()
    )

    return security_set_yields, model_yields

#Example usage:
security_set_yields, model_yields = calculate_weighted_dividend_yields('foguth_etf_models.db')
print(security_set_yields)
print(model_yields)
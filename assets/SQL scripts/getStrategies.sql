CREATE VIEW `view1` AS
SELECT 
            sse.security_set_id,
            sse.etf_id,
            sse.weight,
            sse.startDate,
            sse.endDate,
            ep.Date,
            ep.Close AS etf_price
        FROM security_sets_etfs sse
        JOIN etf_prices ep ON sse.etf_id = ep.etf_id
        WHERE ep.Date BETWEEN sse.startDate AND COALESCE(sse.endDate, '9999-12-31')
        ORDER BY sse.security_set_id, ep.Date
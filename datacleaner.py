"""
Data Cleaner / Validator for SQLite tables (e.g. benchmark_prices)

Usage examples:
# report only
python data_cleaner.py --db foguthbenchmarks.db --table benchmark_prices

# report + apply selected fixes and write cleaned table (safe: creates backup and cleaned table)
python data_cleaner.py --db foguthbenchmarks.db --table benchmark_prices --apply drop_duplicates parse_dates drop_negatives interpolate remove_outliers --overwrite

Notes:
- By default writing creates a new table <table>_cleaned_<timestamp>.
- If --overwrite is provided the script will first create a backup table <table>_backup_<timestamp> then replace original.
"""
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_table(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM `{table}`", conn)
    except Exception as e:
        conn.close()
        raise
    conn.close()
    return df

def save_table(df, db_path, table_name, overwrite=False):
    conn = sqlite3.connect(db_path)
    if_exists = 'replace' if overwrite else 'fail'
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()

def make_backup_table(db_path, table):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_name = f"{table}_backup_{ts}"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (backup_name,))
    if cur.fetchone():
        conn.close()
        raise RuntimeError("Backup table already exists (unlikely).")
    cur.execute(f"CREATE TABLE {backup_name} AS SELECT * FROM {table};")
    conn.commit()
    conn.close()
    return backup_name

def detect_issues(df, table):
    report = []
    nrows, ncols = df.shape
    report.append(("Table", table))
    report.append(("Rows", nrows))
    report.append(("Columns", ncols))
    report.append(("Column names", ", ".join(df.columns.tolist())))
    # Null counts
    nulls = df.isnull().sum()
    for col, cnt in nulls.items():
        report.append((f"Nulls in {col}", int(cnt)))
    # dtype summary
    for col in df.columns:
        report.append((f"Dtype {col}", str(df[col].dtype)))
    # Exact duplicates
    dup_rows = df.duplicated().sum()
    report.append(("Exact duplicate rows", int(dup_rows)))
    # Duplicate key candidates common to investment tables
    # check for (strategy_benchmark, date) or (symbol, Date)
    candidate_keys = []
    if 'strategy_benchmark' in df.columns and 'date' in df.columns:
        dups = df.duplicated(subset=['strategy_benchmark', 'date']).sum()
        candidate_keys.append(( "Duplicate (strategy_benchmark,date)", int(dups)))
    if 'symbol' in df.columns and 'Date' in df.columns:
        dups = df.duplicated(subset=['symbol', 'Date']).sum()
        candidate_keys.append(( "Duplicate (symbol,Date)", int(dups)))
    for k,v in candidate_keys:
        report.append((k, v))
    # Date parse issues
    date_cols = [c for c in df.columns if c.lower() in ('date', 'datetime', 'day')]
    parse_issues = {}
    for dc in date_cols:
        try:
            parsed = pd.to_datetime(df[dc], errors='coerce')
            bad = parsed.isna().sum()
            parse_issues[dc] = int(bad)
            report.append((f"Unparseable dates in {dc}", int(bad)))
        except Exception as e:
            report.append((f"Date parse error {dc}", str(e)))
    # Numeric columns summary: negative / zero counts / outliers by z-score
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for nc in num_cols:
        col = df[nc].dropna().astype(float)
        neg = (col < 0).sum()
        zeros = (col == 0).sum()
        mean = float(col.mean()) if len(col)>0 else float('nan')
        std = float(col.std()) if len(col)>0 else float('nan')
        report.append((f"Numeric {nc} mean", mean))
        report.append((f"Numeric {nc} std", std))
        report.append((f"Numeric {nc} negatives", int(neg)))
        report.append((f"Numeric {nc} zeros", int(zeros)))
        # outliers via zscore > 4
        if len(col) >= 5 and not math.isnan(std) and std > 0:
            z = (col - mean) / std
            out = (z.abs() > 4).sum()
            report.append((f"Numeric {nc} outliers(|z|>4)", int(out)))
    # Grouped date gaps: per strategy or symbol, look for missing trading days
    # only attempt when date column exists and parseable
    if date_cols:
        dc = date_cols[0]
        parsed = pd.to_datetime(df[dc], errors='coerce')
        if parsed.isna().sum() == 0:
            df['_parsed_date_'] = parsed
            group_col = None
            if 'strategy_benchmark' in df.columns:
                group_col = 'strategy_benchmark'
            elif 'symbol' in df.columns:
                group_col = 'symbol'
            if group_col:
                # compute max gap between consecutive days in calendar days per group
                gaps = {}
                for name, g in df.groupby(group_col):
                    idx = g['_parsed_date_'].dropna().sort_values().unique()
                    if len(idx) < 2:
                        continue
                    diffs = np.diff(idx.astype('datetime64[D]').astype(int))
                    maxgap = int(diffs.max())
                    gaps[name] = maxgap
                # find groups with gap > 7 (week) or > 3 trading days
                big_gaps = {k:v for k,v in gaps.items() if v > 7}
                report.append((f"Groups with calendar max gap >7 days ({group_col})", len(big_gaps)))
            df.drop(columns=['_parsed_date_'], inplace=True, errors=True)
    return pd.DataFrame(report, columns=['Check','Value'])

def drop_exact_duplicates(df):
    before = len(df)
    df2 = df.drop_duplicates()
    return df2, before-len(df2)

def coerce_date_cols(df, prefer_col=None, date_format=None):
    # Attempt to locate date column(s) and coerce to 'date' or 'Date' column in dataframe
    candidates = [c for c in df.columns if c.lower() in ('date', 'datetime', 'day')]
    if prefer_col and prefer_col in df.columns:
        candidates = [prefer_col] + [c for c in candidates if c!=prefer_col]
    if not candidates:
        return df, []
    fixed = []
    for c in candidates:
        parsed = pd.to_datetime(df[c], errors='coerce', format=date_format)
        num_bad = int(parsed.isna().sum())
        df[c] = parsed
        fixed.append((c, num_bad))
    # if 'date' not present but 'Date' variant is, rename the first parsed candidate to 'Date'
    if 'Date' not in df.columns and candidates:
        df.rename(columns={candidates[0]:'Date'}, inplace=True)
    return df, fixed

def drop_or_mark_negatives(df, price_cols=None, action='mark'):
    # action: 'drop' or 'mark' (set to NaN)
    if price_cols is None:
        # guess price columns by name
        candidates = [c for c in df.columns if c.lower() in ('close_price','close','price','adjclose','adj_close')]
    else:
        candidates = price_cols
    changed = []
    for c in candidates:
        if c not in df.columns:
            continue
        mask = df[c].notna() & (df[c] < 0)
        count = int(mask.sum())
        if count > 0:
            if action == 'drop':
                df = df.loc[~mask].copy()
            else:
                df.loc[mask, c] = np.nan
            changed.append((c, count))
    return df, changed

def interpolate_prices(df, group_cols=None, price_col='close_price', method='time'):
    # Interpolate per group
    if price_col not in df.columns:
        return df, 0
    if group_cols is None:
        # choose group as strategy_benchmark or symbol
        if 'strategy_benchmark' in df.columns:
            group_cols = ['strategy_benchmark']
        elif 'symbol' in df.columns:
            group_cols = ['symbol']
        else:
            group_cols = []
    replaced = 0
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values(group_cols + ['Date'], inplace=True, ignore_index=True)
    else:
        # sort by index to avoid naive interpolation mixing values
        df.sort_index(inplace=True)
    for name, g in (df.groupby(group_cols) if group_cols else [('__all__', df)]):
        idx = g.index
        before = g[price_col].isna().sum()
        # try interpolation (time if Date exists)
        if 'Date' in g.columns and method == 'time':
            s = g.set_index('Date')[price_col].interpolate(method='time', limit_area='inside')
            df.loc[idx, price_col] = s.values
        else:
            s = g[price_col].interpolate(limit_area='inside')
            df.loc[idx, price_col] = s.values
        after = df.loc[idx, price_col].isna().sum()
        replaced += (before - after)
    return df, int(replaced)

def remove_outliers_zscore(df, num_cols=None, z_thresh=4, action='mark'):
    # action: 'mark' sets outlier to NaN, 'drop' removes rows
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    total_changed = 0
    for c in num_cols:
        col = df[c]
        mean = col.mean(skipna=True)
        std = col.std(skipna=True)
        if std == 0 or np.isnan(std) or col.isna().all():
            continue
        z = (col - mean).abs() / std
        mask = z > z_thresh
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        total_changed += cnt
        if action == 'drop':
            df = df.loc[~mask].copy()
        else:
            df.loc[mask, c] = np.nan
    return df, int(total_changed)

def dedupe_group_date(df, group='strategy_benchmark', datecol='date', agg='last'):
    # Remove duplicate group/date pairs. agg: 'last' or 'mean'
    if group not in df.columns or datecol not in df.columns:
        return df, 0
    # ensure datecol is datetime
    parsed = pd.to_datetime(df[datecol], errors='coerce')
    df['_parsed_date_'] = parsed
    before = len(df)
    if agg == 'last':
        idx = df.sort_values('_parsed_date_').groupby([group, '_parsed_date_']).tail(1).index
        df2 = df.loc[idx].drop(columns=['_parsed_date_']).reset_index(drop=True)
    else:
        # group and take mean for numeric columns, keep first non-null for strings
        def combine(g):
            numeric = g.select_dtypes(include=[np.number]).mean()
            nonnum = g.select_dtypes(exclude=[np.number]).ffill().bfill().iloc[0]
            combined = pd.concat([nonnum, numeric])
            return combined
        df2 = df.groupby([group, '_parsed_date_']).apply(combine).reset_index(drop=True)
    changed = before - len(df2)
    df2.reset_index(drop=True, inplace=True)
    return df2, int(changed)

def generate_report_csv(report_df, db_path, table):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{table}_data_issue_report_{ts}.csv"
    report_df.to_csv(fname, index=False)
    return fname

def main():
    parser = argparse.ArgumentParser(description="Validate and clean investment tables from SQLite DB")
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--table", required=True, help="Table name to inspect/clean")
    parser.add_argument("--apply", nargs='*', choices=['drop_duplicates','parse_dates','drop_negatives','interpolate','remove_outliers','dedupe_group_date'], help="Fixes to apply")
    parser.add_argument("--overwrite", action='store_true', help="If set, overwrite original table after backup (dangerous if used unintentionally)")
    parser.add_argument("--out-name", help="Optional output table name (default: <table>_cleaned_<ts>)")
    parser.add_argument("--z-thresh", type=float, default=4.0, help="Z-score threshold for outlier detection")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    db_path = args.db
    table = args.table

    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        sys.exit(2)

    print(f"Loading table '{table}' from {db_path} ...")
    try:
        df = load_table(db_path, table)
    except Exception as e:
        print("Failed to load table:", e)
        sys.exit(1)

    print(f"Table loaded: {len(df)} rows, {len(df.columns)} columns\n")
    # Run detection
    report_df = detect_issues(df.copy(), table)
    print("===== Data Quality Report =====")
    print(report_df.to_string(index=False))
    csvfile = generate_report_csv(report_df, db_path, table)
    print(f"\nSaved report to {csvfile}")

    if not args.apply:
        print("\nNo fixes requested (--apply not provided). Exiting (report-only mode).")
        return

    # Prepare to apply fixes
    print("\nApplying requested fixes:", args.apply)
    # Make backup of original table
    try:
        backup_name = make_backup_table(db_path, table)
        print(f"Backup of original table created: {backup_name}")
    except Exception as e:
        print("Could not create backup of table:", e)
        print("Aborting to avoid accidental data loss.")
        sys.exit(1)

    cleaned = df.copy()
    # Apply operations in a sensible order
    if 'drop_duplicates' in args.apply:
        cleaned, ndup = drop_exact_duplicates(cleaned)
        print(f"Dropped {ndup} exact duplicate rows")

    if 'parse_dates' in args.apply:
        cleaned, parsed_info = coerce_date_cols(cleaned)
        for col, bad in parsed_info:
            print(f"Coerced date column '{col}', unparseable count: {bad}")

    if 'drop_negatives' in args.apply:
        cleaned, changes = drop_or_mark_negatives(cleaned, action='mark')
        print("Negative price handling (marked to NaN):", changes)

    if 'interpolate' in args.apply:
        # pick a price column
        price_col = None
        for cand in ['close_price','close','Close','price','adjclose','adj_close']:
            if cand in cleaned.columns:
                price_col = cand
                break
        if price_col is None:
            print("No price column found for interpolation. Skipping interpolate.")
        else:
            cleaned, replaced = interpolate_prices(cleaned, price_col=price_col)
            print(f"Interpolated {replaced} missing {price_col} values (group-wise)")

    if 'remove_outliers' in args.apply:
        numcols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        cleaned, removed = remove_outliers_zscore(cleaned, num_cols=numcols, z_thresh=args.z_thresh, action='mark')
        print(f"Marked {removed} numeric values as NaN due to z-score > {args.z_thresh}")

    if 'dedupe_group_date' in args.apply:
        # detect date column name
        datecol = None
        for c in cleaned.columns:
            if c.lower() == 'date':
                datecol = c
                break
        if not datecol:
            # try other likely names
            for c in cleaned.columns:
                if 'date' in c.lower():
                    datecol = c
                    break
        cleaned, changed = dedupe_group_date(cleaned, group='strategy_benchmark' if 'strategy_benchmark' in cleaned.columns else ('symbol' if 'symbol' in cleaned.columns else None), datecol=datecol if datecol else 'date', agg='last')
        print(f"Resolved {changed} duplicate group/date rows using agg='last'")

    # final report of cleaned
    print("\nFinal quick stats after cleaning:")
    final_report = detect_issues(cleaned, table + "_cleaned_preview")
    print(final_report.to_string(index=False))

    # choose output table name
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out_table = args.out_name if args.out_name else f"{table}_cleaned_{ts}"

    # write cleaned table
    try:
        # write cleaned table as new table (fail if exists)
        save_table(cleaned, db_path, out_table, overwrite=False)
        print(f"\nCleaned table written to database as '{out_table}' (did NOT overwrite original).")
        if args.overwrite:
            # replace original (after backup already made)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table};")
            conn.commit()
            conn.close()
            # now write cleaned as original name
            save_table(cleaned, db_path, table, overwrite=False)
            print(f"Original table '{table}' overwritten with cleaned data (original backed up as '{backup_name}').")
    except Exception as e:
        print("Failed to write cleaned table to database:", e)
        print("You can still find the cleaned data in memory or re-run with different options.")
        sys.exit(1)

    print("\nDone. Recommendations:")
    print("- Inspect the backup table created ({}).".format(backup_name))
    print("- Review the cleaned table '{}' before using it in production.".format(out_table))
    print("- If you are happy, replace original with --overwrite next time or run additional fixes.".format(out_table))

if __name__ == "__main__":
    main()
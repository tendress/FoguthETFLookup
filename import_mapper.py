"""
Simple Tkinter Import + Column Mapper (editable full grid)

- Open CSV or Excel
- Auto-suggest mapping to the target fields
- Display full mapped results in an editable grid (double-click to edit a cell)
- Edits are written back to the in-memory mapped DataFrame
- Export mapped CSV

Dependencies:
- pandas
- openpyxl (for .xlsx/.xls)

Run:
python import_mapper.py
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from difflib import get_close_matches
import re
import os

TARGET_COLUMNS = [
    "first_name",
    "last_name",
    "home_phone",
    "email",
    "address",
    "city",
    "state",
    "postal",
    "notes",
    "household_name",
    "household_role",
]

EMAIL_RE = re.compile(r"^[^@]+@[^@]+\.[^@]+$")


def load_file(path):
    name = path.lower()
    if name.endswith((".xls", ".xlsx")):
        # read first sheet by default
        xls = pd.read_excel(path, sheet_name=None)
        # choose first sheet
        first = list(xls.keys())[0]
        df = xls[first]
    else:
        df = pd.read_csv(path)
    return df


def best_match(name, candidates):
    # case-insensitive match and a fuzzy match fallback
    if not candidates:
        return None
    lower_map = {c.lower(): c for c in candidates}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def auto_suggest(source_cols):
    suggestions = {}
    for t in TARGET_COLUMNS:
        # try common synonyms
        synonyms = [
            t, t.replace('_', ' '), t.replace('first', 'fname').replace('last', 'lname'),
            t.capitalize(), t.upper()
        ]
        found = None
        for s in synonyms:
            m = best_match(s, source_cols)
            if m:
                found = m
                break
        if not found:
            # fallback fuzzy match
            found = best_match(t, source_cols)
        suggestions[t] = found
    return suggestions


def validate_mapped_df(df):
    checks = []
    # missing counts
    for c in TARGET_COLUMNS:
        if c in df.columns:
            checks.append((f"Missing in {c}", int(df[c].isna().sum())))
        else:
            checks.append((f"Missing column {c}", "Not mapped"))
    # Emails
    if 'email' in df.columns:
        n_bad = (~df['email'].dropna().astype(str).str.match(EMAIL_RE)).sum()
        checks.append(("Invalid email count", int(n_bad)))
    # Phone digits
    if 'home_phone' in df.columns:
        def digits_only(v):
            if pd.isna(v): return ""
            return re.sub(r"\D", "", str(v))
        phone_digits = df['home_phone'].apply(digits_only)
        too_short = (phone_digits.str.len() < 7) & (phone_digits.str.len() > 0)
        checks.append(("Phone values with <7 digits", int(too_short.sum())))
    # Duplicates heuristic: first+last+email
    if all(col in df.columns for col in ('first_name','last_name','email')):
        dup = df.duplicated(subset=['first_name','last_name','email']).sum()
        checks.append(("Duplicates (first+last+email)", int(dup)))
    # Basic length outliers for postal
    if 'postal' in df.columns:
        non_blank = df['postal'].dropna().astype(str)
        long = (non_blank.str.len() > 12).sum()
        checks.append(("Postal codes longer than 12 chars", int(long)))
    return checks


class ImportMapperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Import & Column Mapper")
        self.geometry("1200x800")
        self.df = None  # source df
        self.mapped_df = None  # mapped df that backs the grid
        self.source_cols = []
        self.mapping_vars = {}  # target -> StringVar for combobox
        self.suggestions = {}
        self.iid_to_index = {}  # map tree iid -> original df index
        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=8, pady=6)

        open_btn = ttk.Button(top_frame, text="Open CSV / Excel", command=self.open_file)
        open_btn.pack(side='left')

        self.file_label = ttk.Label(top_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=8)

        preview_btn = ttk.Button(top_frame, text="Refresh Grid", command=self.on_preview)
        preview_btn.pack(side='right')

        export_btn = ttk.Button(top_frame, text="Export CSV", command=self.on_export)
        export_btn.pack(side='right', padx=8)

        main_panes = ttk.PanedWindow(self, orient='horizontal')
        main_panes.pack(fill='both', expand=True, padx=8, pady=(0,8))

        left_frame = ttk.Frame(main_panes, width=360)
        main_panes.add(left_frame, weight=0)

        right_frame = ttk.Frame(main_panes)
        main_panes.add(right_frame, weight=1)

        # Left: source columns and mapping controls
        ttk.Label(left_frame, text="Source Columns:").pack(anchor='w', padx=6, pady=(6,0))
        self.src_listbox = tk.Listbox(left_frame, height=12, exportselection=False)
        self.src_listbox.pack(fill='x', padx=6, pady=4)

        mapping_frame = ttk.LabelFrame(left_frame, text="Mappings")
        mapping_frame.pack(fill='both', expand=True, padx=6, pady=6)

        # Mapping entries (one combobox per target column)
        self.mapping_widgets = {}
        for t in TARGET_COLUMNS:
            row = ttk.Frame(mapping_frame)
            row.pack(fill='x', padx=4, pady=2)
            lbl = ttk.Label(row, text=t, width=18)
            lbl.pack(side='left')
            var = tk.StringVar()
            cb = ttk.Combobox(row, textvariable=var, values=["(none)"], state='readonly')
            cb.pack(side='left', fill='x', expand=True)
            self.mapping_vars[t] = var
            self.mapping_widgets[t] = cb

        auto_btn = ttk.Button(left_frame, text="Auto-suggest mappings", command=self.auto_map)
        auto_btn.pack(side='left', padx=6, pady=6)

        apply_btn = ttk.Button(left_frame, text="Apply mapping & validate", command=self.on_preview)
        apply_btn.pack(side='right', padx=6, pady=6)

        # Right: preview and validation
        preview_top = ttk.Frame(right_frame)
        preview_top.pack(fill='x')

        self.rows_label = ttk.Label(preview_top, text="No data loaded")
        self.rows_label.pack(anchor='w', padx=6, pady=6)

        # Treeview for full mapped results, editable
        tree_frame = ttk.Frame(right_frame)
        tree_frame.pack(fill='both', expand=True, padx=6, pady=6)
        self.tree = ttk.Treeview(tree_frame, columns=[], show='headings')
        self.tree.pack(side='left', fill='both', expand=True)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        hsb = ttk.Scrollbar(right_frame, orient="horizontal", command=self.tree.xview)
        hsb.pack(fill='x')
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Bind double-click for editing
        self.tree.bind("<Double-1>", self.on_double_click)

        # Validation text
        val_frame = ttk.LabelFrame(right_frame, text="Validation Report")
        val_frame.pack(fill='x', padx=6, pady=6)
        self.val_text = tk.Text(val_frame, height=8, wrap='word')
        self.val_text.pack(fill='both', padx=6, pady=6)

    def open_file(self):
        path = filedialog.askopenfilename(title="Select CSV or Excel", filetypes=[("CSV files","*.csv"),("Excel files","*.xlsx;*.xls")])
        if not path:
            return
        try:
            df = load_file(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not read file: {e}")
            return
        self.df = df
        self.source_cols = list(df.columns.astype(str))
        self.file_label.config(text=os.path.basename(path))
        # populate listbox
        self.src_listbox.delete(0, tk.END)
        for c in self.source_cols:
            self.src_listbox.insert(tk.END, c)
        # populate combobox options
        options = ["(none)"] + self.source_cols
        for t, cb in self.mapping_widgets.items():
            cb['values'] = options
            cb.set("(none)")
        # auto-suggest
        self.suggestions = auto_suggest(self.source_cols)
        # set suggestions as default if present
        for t, suggestion in self.suggestions.items():
            if suggestion:
                self.mapping_vars[t].set(suggestion)

        # clear preview and report
        self.clear_tree()
        self.val_text.delete('1.0', tk.END)
        self.rows_label.config(text=f"Loaded {len(self.df)} rows")

    def auto_map(self):
        if not self.source_cols:
            messagebox.showinfo("No file", "Load a file first")
            return
        self.suggestions = auto_suggest(self.source_cols)
        for t, suggestion in self.suggestions.items():
            if suggestion:
                self.mapping_vars[t].set(suggestion)

    def build_mapped_df(self):
        """
        Build the mapped DataFrame from the loaded source DataFrame according to the
        user mapping. Also ensure the `household_name` column in the preview/export
        is formatted as 'last_name, first_name' when both name parts are available.
        """
        if self.df is None:
            return None

        # Build rename map: source_column -> target_name
        rename = {}
        for t, var in self.mapping_vars.items():
            v = var.get()
            if v and v != "(none)":
                rename[v] = t

        # Rename the DataFrame columns according to mapping
        mapped = self.df.rename(columns=rename).copy()

        # Ensure household_name is 'last_name, first_name' for preview and export.
        # If both last_name and first_name exist, build household_name from them and override any mapped value.
        # If one or both name parts are missing, keep existing household_name (if any).
        if 'last_name' in mapped.columns and 'first_name' in mapped.columns:
            # Create clean string parts (strip whitespace, treat NaN as empty)
            last = mapped['last_name'].fillna('').astype(str).str.strip()
            first = mapped['first_name'].fillna('').astype(str).str.strip()

            # Combine into "Last, First" but avoid stray commas when first is empty
            combined = last.where(last != '', '')  # keep empty strings
            combined = combined + (', ' + first).where(first != '', '')
            # Remove leading/trailing whitespace and convert empty string to NA
            combined = combined.str.strip().replace('', pd.NA)

            mapped['household_name'] = combined
        else:
            # If names not present but household_name was mapped from source, ensure it's string type
            if 'household_name' in mapped.columns:
                mapped['household_name'] = mapped['household_name'].astype('string')

        # store mapped_df for grid editing
        # reset index to preserve original index label in mapping; keep original index in column _orig_index
        mapped = mapped.reset_index(drop=False)
        mapped.rename(columns={'index': '_orig_index'}, inplace=True)
        self.mapped_df = mapped
        return mapped

    def clear_tree(self):
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = []
        self.iid_to_index = {}

    def populate_tree(self, df_full):
        """
        Populate the tree with the DataFrame df_full but only show the canonical
        target columns in the preferred order. '_orig_index' is kept for internal
        mapping but is not shown. Ensures each target column exists (adds
        pd.NA for missing columns) so they are editable in the grid.
        """
        self.clear_tree()
        if df_full is None or df_full.empty:
            return

        # Columns we want to display (in this specific order)
        display_order = [
            "first_name", "last_name", "home_phone", "email",
            "address", "city", "state", "postal",
            "notes", "household_name", "household_role"
        ]

        # Work on a copy to ensure we don't mutate caller's df unexpectedly
        df_copy = df_full.copy()

        # Ensure all target columns exist in the DataFrame so the grid shows them (even if empty)
        for col in display_order:
            if col not in df_copy.columns:
                df_copy[col] = pd.NA

        # Ensure internal index column exists so editing logic can map updates back
        if '_orig_index' not in df_copy.columns:
            df_copy = df_copy.reset_index(drop=False)
            df_copy.rename(columns={'index': '_orig_index'}, inplace=True)

        # Keep the updated mapped_df with all target columns (editable source)
        self.mapped_df = df_copy

        # Display only the target columns (in display_order)
        cols = display_order

        # set up tree columns
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor='w')

        # insert all rows
        for i, row in df_copy.iterrows():
            iid = f"r{i}"
            vals = [self._fmt_cell(row.get(c)) for c in cols]
            self.tree.insert("", "end", iid=iid, values=vals)
            # Keep mapping from tree iid to original index value for editing
            self.iid_to_index[iid] = row.get('_orig_index', i)

        self.rows_label.config(text=f"Showing {len(df_copy)} rows")

    def _fmt_cell(self, v):
        if pd.isna(v):
            return ""
        return str(v)

    def on_preview(self):
        mapped = self.build_mapped_df()
        if mapped is None:
            messagebox.showinfo("No file", "Load a file first")
            return
        # Show full dataframe in grid
        self.populate_tree(mapped)
        # Run validation
        checks = validate_mapped_df(mapped.rename(columns={'_orig_index': 'orig_index'}))
        self.val_text.delete('1.0', tk.END)
        out_lines = []
        for k,v in checks:
            out_lines.append(f"{k}: {v}")
        self.val_text.insert('1.0', "\n".join(out_lines))

    def on_export(self):
        """
        Export only the columns currently displayed in the grid (and in that order).
        Uses the current in-memory mapped_df (which reflects any edits made in the grid).
        """
        if self.mapped_df is None:
            messagebox.showinfo("No data", "Build mapping first")
            return

        # Offer filename
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")], title="Save mapped CSV as")
        if not f:
            return
        try:
            # Columns currently displayed in the grid (order preserved)
            grid_cols = list(self.tree["columns"])

            # Build export DataFrame from the in-memory mapped_df so it contains edits
            export_df = self.mapped_df.copy()

            # Drop internal helper column if present
            if '_orig_index' in export_df.columns:
                export_df = export_df.drop(columns=['_orig_index'])

            # Ensure export_df contains all columns that are in grid_cols (add missing ones as NA)
            for c in grid_cols:
                if c not in export_df.columns:
                    export_df[c] = pd.NA

            # Select only the grid columns in the same order
            final = export_df[grid_cols]

            # Write to CSV
            final.to_csv(f, index=False)
            messagebox.showinfo("Saved", f"Saved mapped CSV to: {f}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # Editing support: double-click to edit a cell in-place
    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_iid = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)  # e.g. '#3'
        if not row_iid or not col:
            return
        col_index = int(col.replace('#', '')) - 1
        col_name = self.tree["columns"][col_index]
        # get bbox for the cell
        bbox = self.tree.bbox(row_iid, column=col_name)
        if not bbox:
            return
        x, y, width, height = bbox
        # current cell value
        cur_val = self.tree.set(row_iid, col_name)
        # create entry widget overlay
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, cur_val)
        entry.focus_set()

        def save_edit(event=None):
            new_val = entry.get()
            entry.destroy()
            # update treeview cell
            self.tree.set(row_iid, col_name, new_val)
            # update mapped_df
            mapped_index = self.iid_to_index.get(row_iid)
            if mapped_index is None:
                return
            # find row in self.mapped_df where _orig_index == mapped_index
            mask = self.mapped_df['_orig_index'] == mapped_index
            if not mask.any():
                # fallback: maybe original index is numeric i
                try:
                    # try parse iid numeric part
                    idx_num = int(row_iid.replace('r',''))
                    if idx_num < len(self.mapped_df):
                        self.mapped_df.at[idx_num, col_name] = new_val
                except Exception:
                    pass
            else:
                # set the value
                self.mapped_df.loc[mask, col_name] = new_val
            # If first_name or last_name changed, recompute household_name
            if col_name in ('first_name', 'last_name'):
                self._recompute_household_for_index(mapped_index)

        def cancel_edit(event=None):
            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)
        entry.bind("<Escape>", cancel_edit)

    def _recompute_household_for_index(self, mapped_index):
        """
        Recompute household_name for the row(s) with given original index.
        """
        mask = self.mapped_df['_orig_index'] == mapped_index
        if not mask.any():
            return
        for idx in self.mapped_df[mask].index:
            first = str(self.mapped_df.at[idx, 'first_name']) if 'first_name' in self.mapped_df.columns else ''
            last = str(self.mapped_df.at[idx, 'last_name']) if 'last_name' in self.mapped_df.columns else ''
            first = first.strip() if first not in (None, pd.NA) else ''
            last = last.strip() if last not in (None, pd.NA) else ''
            if last and first:
                combined = f"{last}, {first}"
            elif last:
                combined = last
            elif first:
                combined = first
            else:
                combined = pd.NA
            self.mapped_df.at[idx, 'household_name'] = combined
            # update tree cell visually
            # find iid for this row
            # stored mapping may have multiple iids mapping to same orig_index; update all
            for iid, orig in self.iid_to_index.items():
                if orig == mapped_index:
                    if 'household_name' in self.tree["columns"]:
                        self.tree.set(iid, 'household_name', "" if pd.isna(combined) else str(combined))

if __name__ == "__main__":
    app = ImportMapperApp()
    app.mainloop()
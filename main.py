import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Logic: Excel Loader
# ---------------------------
class ExcelLoaderThread(threading.Thread):
    def __init__(self, path, on_progress, on_done, on_error):
        super().__init__(daemon=True)
        self.path = path
        self.on_progress = on_progress
        self.on_done = on_done
        self.on_error = on_error

    def run(self):
        try:
            self.on_progress(10, "Reading Excel...")
            df = pd.read_excel(self.path, engine='openpyxl')
            
            self.on_progress(30, "Cleaning data...")
            if 'Name' in df.columns:
                df = df.drop_duplicates(subset='Name', keep='first')
            df = df.fillna('')
            target_cols = ['Name', 'Tags', 'Genres', 'Categories', 'Developers', 'About the game', 'Publishers', 'Price', 'Release date']
            for col in target_cols:
                if col not in df.columns:
                    df[col] = ''
                df[col] = df[col].astype(str)
            
            self.on_progress(60, "Combining features...")
            df['combined_features'] = (
                df['Tags'] + " " + df['Genres'] + " " + 
                df['Categories'] + " " + df['Developers']
            )
            
            self.on_done(df)
        except Exception as e:
            self.on_error(e)

# ---------------------------
# Logic: Recommender Engine
# ---------------------------
def build_tfidf_matrix(df, on_progress_callback=None):
    if on_progress_callback:
        on_progress_callback(80, "Training AI model...")
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    return tfidf.fit_transform(df['combined_features'])

def recommend_from_user_ratings(user_ratings, df, tfidf_matrix, top_n=5):
    if not user_ratings: return []
    
    # Setup lookup
    df['name_lower'] = df['Name'].str.lower().str.strip()
    name_to_idx = {name: i for i, name in enumerate(df['name_lower'])}
    
    # Build User Profile
    user_profile = None
    for name, rating in user_ratings.items():
        clean_name = name.lower().strip()
        idx = name_to_idx.get(clean_name)
        if idx is not None:
            vec = tfidf_matrix[idx]
            weight = rating - 2.5 # Weighting strategy
            if user_profile is None:
                user_profile = vec * weight
            else:
                user_profile = user_profile + (vec * weight)
    
    if user_profile is None: return []
    
    # Calculate & Scale
    sims = cosine_similarity(user_profile, tfidf_matrix).ravel()
    
    # Exclude played
    for played in user_ratings.keys():
        idx = name_to_idx.get(played.lower().strip())
        if idx is not None: sims[idx] = -1.0
            
    # Get Top N
    top_idx = np.argsort(sims)[::-1][:top_n]
    top_scores = sims[top_idx]
    
    # Scaling to percentages
    if len(top_scores) > 0 and top_scores[0] > 0:
        scaler = MinMaxScaler(feature_range=(75, 98))
        scaled = scaler.fit_transform(top_scores.reshape(-1, 1)).flatten()
    else:
        scaled = top_scores * 100

    recs = []
    for i, idx in enumerate(top_idx):
        if sims[idx] <= 0: continue
        row = df.iloc[idx]
        recs.append({
            "Name": row["Name"],
            "Genres": row["Genres"],
            "Release": row["Release date"],
            "Score": scaled[i]
        })
    return recs

# ---------------------------
# UI Components
# ---------------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Steam Games Recommender System By 23k0720 and 23k0612")
        root.geometry("1200x800")
        
        self.df = None
        self.tfidf = None
        self.ratings = {}

        main_pane = ttk.PanedWindow(root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        # 1. Loader
        load_grp = ttk.LabelFrame(left_frame, text="1. Load Data", padding=5)
        load_grp.pack(fill="x", pady=5)
        
        self.path_var = tk.StringVar()
        ttk.Entry(load_grp, textvariable=self.path_var).pack(fill="x", pady=2)
        btn_frm = ttk.Frame(load_grp)
        btn_frm.pack(fill="x")
        ttk.Button(btn_frm, text="Browse", command=self.browse).pack(side="left", expand=True, fill="x")
        self.btn_load = ttk.Button(btn_frm, text="Load", command=self.load_data, state="disabled")
        self.btn_load.pack(side="left", expand=True, fill="x")
        
        self.prog = ttk.Progressbar(load_grp, mode="determinate")
        self.prog.pack(fill="x", pady=5)
        self.lbl_status = ttk.Label(load_grp, text="Waiting for file...", font=("Arial", 8))
        self.lbl_status.pack(anchor="w")

        # 2. Selector
        sel_grp = ttk.LabelFrame(left_frame, text="2. Select & Rate", padding=5)
        sel_grp.pack(fill="both", expand=True, pady=5)
        
        self.search_box = SearchableListbox(sel_grp)
        self.search_box.pack(fill="both", expand=True)
        self.search_box.bind_select(self.on_game_click)
        
        # Rating Controls
        rate_frm = ttk.Frame(sel_grp)
        rate_frm.pack(fill="x", pady=5)
        ttk.Label(rate_frm, text="Rate:").pack(side="left")
        self.spin_rate = tk.Spinbox(rate_frm, from_=1, to=5, width=3)
        self.spin_rate.delete(0, "end"); self.spin_rate.insert(0, 5)
        self.spin_rate.pack(side="left", padx=5)
        
        self.btn_add = ttk.Button(rate_frm, text="Add", command=self.add_rate, state="disabled")
        self.btn_add.pack(side="left", fill="x", expand=True)
        self.btn_rem = ttk.Button(rate_frm, text="Del", command=self.del_rate, state="disabled")
        self.btn_rem.pack(side="left", fill="x", expand=True)

        # 3. User Ratings List
        rate_list_grp = ttk.LabelFrame(left_frame, text="Your Profile", padding=5)
        rate_list_grp.pack(fill="both", expand=True, pady=5)
        self.lst_ratings = tk.Listbox(rate_list_grp, height=8)
        self.lst_ratings.pack(fill="both", expand=True)

        # 4. Action Button
        self.btn_rec = ttk.Button(left_frame, text="GET RECOMMENDATIONS ➤", command=self.calc_recs, state="disabled")
        self.btn_rec.pack(fill="x", pady=10, ipady=5)
        
        # 1. Detailed Info
        info_grp = ttk.LabelFrame(right_frame, text="Game Details", padding=10)
        info_grp.pack(fill="both", expand=True, pady=5)
        
        # Using a Text widget with tags for formatting
        self.txt_info = tk.Text(info_grp, wrap="word", font=("Segoe UI", 10), state="disabled", padx=10, pady=10)
        sb_info = ttk.Scrollbar(info_grp, command=self.txt_info.yview)
        self.txt_info.configure(yscrollcommand=sb_info.set)
        sb_info.pack(side="right", fill="y")
        self.txt_info.pack(fill="both", expand=True)
        
        # Tags for formatting
        self.txt_info.tag_config("title", font=("Segoe UI", 16, "bold"), foreground="#2563eb", spacing3=10)
        self.txt_info.tag_config("meta_key", font=("Segoe UI", 9, "bold"), foreground="#4b5563")
        self.txt_info.tag_config("meta_val", font=("Segoe UI", 9), foreground="#111827")
        self.txt_info.tag_config("section", font=("Segoe UI", 11, "bold", "underline"), spacing1=15, spacing3=5)
        self.txt_info.tag_config("body", font=("Segoe UI", 10), spacing1=2)

        # 2. Recommendations
        rec_grp = ttk.LabelFrame(right_frame, text="Top 5 Recommendations", padding=10)
        rec_grp.pack(fill="both", expand=True, pady=5)
        
        self.txt_recs = tk.Text(rec_grp, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.txt_recs.pack(fill="both", expand=True)
        self.txt_recs.tag_config("rec_title", font=("Segoe UI", 12, "bold"), foreground="#16a34a")
        self.txt_recs.tag_config("rec_meta", font=("Segoe UI", 9), foreground="#6b7280")

    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if p:
            self.path_var.set(p)
            self.btn_load.config(state="normal")

    def load_data(self):
        self.btn_load.config(state="disabled")
        ExcelLoaderThread(self.path_var.get(), self.update_prog, self.done_loading, self.show_err).start()

    def update_prog(self, val, msg):
        self.root.after(0, lambda: [self.prog.configure(value=val), self.lbl_status.configure(text=msg)])

    def done_loading(self, df):
        self.root.after(0, lambda: self._finalize_load(df))

    def _finalize_load(self, df):
        self.df = df
        self.tfidf = build_tfidf_matrix(df, self.update_prog)
        self.update_prog(100, "Ready!")
        
        self.search_box.set_items(df['Name'].unique())
        for b in [self.btn_add, self.btn_rem, self.btn_rec]: b.config(state="normal")

    def show_err(self, e):
        messagebox.showerror("Error", str(e))

    def on_game_click(self):
        name = self.search_box.get_selection()
        if not name or self.df is None: return
        
        row = self.df[self.df['Name'] == name].iloc[0]
        
        self.txt_info.config(state="normal")
        self.txt_info.delete(1.0, "end")
        
        # Insert Title
        self.txt_info.insert("end", f"{row['Name']}\n", "title")
        
        # Insert Metadata Table
        meta = [
            ("Release Date", row['Release date']),
            ("Price", row['Price']),
            ("Developers", row['Developers']),
            ("Publishers", row['Publishers']),
            ("Genres", row['Genres']),
            ("Tags", row['Tags']),
        ]
        
        for k, v in meta:
            self.txt_info.insert("end", f"{k}: ", "meta_key")
            self.txt_info.insert("end", f"{v}\n", "meta_val")
            
        # Insert Description
        self.txt_info.insert("end", "\nAbout this game\n", "section")
        self.txt_info.insert("end", row['About the game'], "body")
        
        self.txt_info.config(state="disabled")

    def add_rate(self):
        name = self.search_box.get_selection()
        if name:
            self.ratings[name] = float(self.spin_rate.get())
            self.refresh_list()

    def del_rate(self):
        sel = self.lst_ratings.curselection()
        if sel:
            name = self.lst_ratings.get(sel[0]).split(" | ")[0]
            del self.ratings[name]
            self.refresh_list()

    def refresh_list(self):
        self.lst_ratings.delete(0, "end")
        for n, r in self.ratings.items():
            self.lst_ratings.insert("end", f"{n} | {r}⭐")

    def calc_recs(self):
        if not self.ratings:
            messagebox.showwarning("Wait", "Rate at least one game first!")
            return
            
        self.lbl_status.config(text="Thinking...")
        self.root.update()
        
        recs = recommend_from_user_ratings(self.ratings, self.df, self.tfidf, top_n=5)
        
        self.txt_recs.config(state="normal")
        self.txt_recs.delete(1.0, "end")
        
        for i, r in enumerate(recs, 1):
            self.txt_recs.insert("end", f"{i}. {r['Name']} ", "rec_title")
            self.txt_recs.insert("end", f"({r['Score']:.1f}% Match)\n", "rec_title")
            self.txt_recs.insert("end", f"   Released: {r['Release']} | Genres: {r['Genres']}\n\n", "rec_meta")
            
        self.txt_recs.config(state="disabled")
        self.lbl_status.config(text="Done.")


class SearchableListbox(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_type_debounced)
        self._after_id = None
        
        ttk.Entry(self, textvariable=self.search_var).pack(fill="x", padx=2, pady=2)
        
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)
        
        sb = ttk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        
        self.listbox = tk.Listbox(frame, yscrollcommand=sb.set, height=10)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        
        self.all_items = []
        self.callback = None
        self.listbox.bind('<<ListboxSelect>>', lambda e: self.callback() if self.callback else None)

    def on_type_debounced(self, *args):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(200, self.filter)

    def set_items(self, items):
        self.all_items = sorted(items)
        self.filter()

    def filter(self, *args):
        term = self.search_var.get().lower()
        self.listbox.delete(0, tk.END)
        if term:
            items = [i for i in self.all_items if term in i.lower()]
        else:
            items = self.all_items
        if items:
            self.listbox.insert(tk.END, *items)

    def get_selection(self):
        sel = self.listbox.curselection()
        return self.listbox.get(sel[0]) if sel else None

    def bind_select(self, cb):
        self.callback = cb

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
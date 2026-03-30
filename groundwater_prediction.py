"""
Groundwater Level Prediction Using Hybrid ANN with Genetic Algorithm
=====================================================================
Full implementation with:
  - ANN + Crow Search GA
  - ANN + Grey Wolf GA
  - MSE Comparison
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
dataset       = None
X_train = X_test = y_train = y_test = None
mse_crow = mse_wolf = None


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def log(msg):
    output_box.configure(state="normal")
    output_box.insert(tk.END, str(msg) + "\n")
    output_box.see(tk.END)
    output_box.configure(state="disabled")
    root.update()

def clear_log():
    output_box.configure(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.configure(state="disabled")


# ─────────────────────────────────────────────
# SIMPLE ANN (pure numpy – no TF dependency)
# ─────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

class SimpleANN:
    """
    Single-hidden-layer feedforward ANN trained with back-propagation.
    Weights can be pre-seeded by the GA optimiser.
    """
    def __init__(self, n_input, n_hidden=10, n_output=1, lr=0.01, epochs=300):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr       = lr
        self.epochs   = epochs
        self._init_weights()

    def _init_weights(self):
        self.W1 = np.random.randn(self.n_input,  self.n_hidden) * 0.1
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, self.n_output) * 0.1
        self.b2 = np.zeros((1, self.n_output))

    def set_weights_from_vector(self, vec):
        """Load a flat weight vector produced by the GA."""
        idx = 0
        s1  = self.n_input * self.n_hidden
        self.W1 = vec[idx:idx+s1].reshape(self.n_input, self.n_hidden); idx += s1
        s2  = self.n_hidden
        self.b1 = vec[idx:idx+s2].reshape(1, self.n_hidden);            idx += s2
        s3  = self.n_hidden * self.n_output
        self.W2 = vec[idx:idx+s3].reshape(self.n_hidden, self.n_output); idx += s3
        s4  = self.n_output
        self.b2 = vec[idx:idx+s4].reshape(1, self.n_output)

    def weight_count(self):
        return (self.n_input*self.n_hidden + self.n_hidden +
                self.n_hidden*self.n_output + self.n_output)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2          # linear output for regression

    def fit(self, X, y, verbose=False):
        for ep in range(self.epochs):
            out   = self.forward(X)
            err   = out - y
            dW2   = self.a1.T @ err / len(X)
            db2   = err.mean(axis=0, keepdims=True)
            dA1   = err @ self.W2.T
            dZ1   = dA1 * sigmoid_deriv(self.z1)
            dW1   = X.T @ dZ1 / len(X)
            db1   = dZ1.mean(axis=0, keepdims=True)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        return self.forward(X)


# ─────────────────────────────────────────────
# FITNESS: MSE on training data
# ─────────────────────────────────────────────
def fitness_ann(weights, ann, X, y):
    ann.set_weights_from_vector(weights)
    pred = ann.predict(X)
    return mean_squared_error(y, pred)


# ─────────────────────────────────────────────
# CROW SEARCH ALGORITHM  (meta-heuristic GA)
# ─────────────────────────────────────────────
def crow_search(ann, X_tr, y_tr, n_crows=20, iters=50, AP=0.1, FL=2.0):
    """
    Crow Search Algorithm for optimising ANN weights.
    AP  = Awareness Probability
    FL  = Flight Length
    """
    dim   = ann.weight_count()
    crows = np.random.uniform(-1, 1, (n_crows, dim))
    mem   = crows.copy()
    fit   = np.array([fitness_ann(c, ann, X_tr, y_tr) for c in crows])
    m_fit = fit.copy()

    best_fit = np.min(m_fit)
    best_pos = mem[np.argmin(m_fit)].copy()

    for _ in range(iters):
        for i in range(n_crows):
            j = np.random.randint(n_crows)
            if np.random.rand() >= AP:
                crows[i] = crows[i] + np.random.rand() * FL * (mem[j] - crows[i])
            else:
                crows[i] = np.random.uniform(-1, 1, dim)
            f = fitness_ann(crows[i], ann, X_tr, y_tr)
            if f < m_fit[i]:
                mem[i]   = crows[i].copy()
                m_fit[i] = f
        if np.min(m_fit) < best_fit:
            best_fit = np.min(m_fit)
            best_pos = mem[np.argmin(m_fit)].copy()

    ann.set_weights_from_vector(best_pos)
    return ann, best_fit


# ─────────────────────────────────────────────
# GREY WOLF OPTIMIZER  (meta-heuristic GA)
# ─────────────────────────────────────────────
def grey_wolf(ann, X_tr, y_tr, n_wolves=20, iters=50):
    """Grey Wolf Optimizer for ANN weight optimisation."""
    dim    = ann.weight_count()
    wolves = np.random.uniform(-1, 1, (n_wolves, dim))
    fit    = np.array([fitness_ann(w, ann, X_tr, y_tr) for w in wolves])

    idx_sorted = np.argsort(fit)
    alpha, beta, delta = wolves[idx_sorted[0]].copy(), \
                         wolves[idx_sorted[1]].copy(), \
                         wolves[idx_sorted[2]].copy()
    fa, fb, fd = fit[idx_sorted[0]], fit[idx_sorted[1]], fit[idx_sorted[2]]

    for t in range(iters):
        a = 2 - 2 * t / iters          # linearly decreases from 2 → 0
        for i in range(n_wolves):
            for leader, _ in [(alpha, fa), (beta, fb), (delta, fd)]:
                A = 2 * a * np.random.rand(dim) - a
                C = 2 * np.random.rand(dim)
                wolves[i] -= (A * np.abs(C * leader - wolves[i])) / 3
            f = fitness_ann(wolves[i], ann, X_tr, y_tr)
            fit[i] = f

        idx_sorted = np.argsort(fit)
        if fit[idx_sorted[0]] < fa:
            alpha = wolves[idx_sorted[0]].copy(); fa = fit[idx_sorted[0]]
        if fit[idx_sorted[1]] < fb:
            beta  = wolves[idx_sorted[1]].copy(); fb = fit[idx_sorted[1]]
        if fit[idx_sorted[2]] < fd:
            delta = wolves[idx_sorted[2]].copy(); fd = fit[idx_sorted[2]]

    ann.set_weights_from_vector(alpha)
    return ann, fa


# ─────────────────────────────────────────────
# FEATURE SELECTION (binary GA)
# ─────────────────────────────────────────────
def select_features_ga(X, y, n_pop=20, iters=30):
    """Simple binary GA for feature selection. Returns selected column indices."""
    n_feat = X.shape[1]
    pop    = np.random.randint(0, 2, (n_pop, n_feat))
    # ensure at least 1 feature per individual
    for ind in pop:
        if ind.sum() == 0:
            ind[np.random.randint(n_feat)] = 1

    def feat_fitness(mask):
        cols = np.where(mask == 1)[0]
        if len(cols) == 0:
            return 1e9
        ann_tmp = SimpleANN(len(cols), n_hidden=8, epochs=50)
        ann_tmp.fit(X[:, cols], y)
        pred = ann_tmp.predict(X[:, cols])
        return mean_squared_error(y, pred)

    fit = np.array([feat_fitness(ind) for ind in pop])

    for _ in range(iters):
        # tournament selection
        new_pop = []
        for _ in range(n_pop):
            i, j   = np.random.choice(n_pop, 2, replace=False)
            winner = pop[i] if fit[i] < fit[j] else pop[j]
            child  = winner.copy()
            # mutation
            mut_idx = np.random.randint(n_feat)
            child[mut_idx] = 1 - child[mut_idx]
            if child.sum() == 0:
                child[np.random.randint(n_feat)] = 1
            new_pop.append(child)
        pop = np.array(new_pop)
        fit = np.array([feat_fitness(ind) for ind in pop])

    best = pop[np.argmin(fit)]
    return np.where(best == 1)[0]


# ─────────────────────────────────────────────
# MODULE 1 – UPLOAD DATASET
# ─────────────────────────────────────────────
def upload_dataset():
    global dataset
    path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not path:
        return
    dataset = pd.read_csv(path)
    file_lbl.configure(text=path)
    clear_log()
    log("Dataset loaded successfully.")
    log(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")


# ─────────────────────────────────────────────
# MODULE 2 – PREPROCESS
# ─────────────────────────────────────────────
def preprocess_dataset():
    global dataset, X_train, X_test, y_train, y_test
    if dataset is None:
        log("⚠  Please upload a dataset first."); return

    clear_log()
    df = dataset.copy()

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Label-encode string columns (except target)
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    log("=== Sample Data (first 5 rows) ===")
    log(df.head().to_string())
    log(f"\n[{df.shape[0]} rows × {df.shape[1]} columns]")

    # Last numeric column = target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).values.astype(float)
    y = df[target_col].values.reshape(-1, 1).astype(float)

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    log(f"\nPreprocessing done.")
    log(f"Training samples : {len(X_train)}")
    log(f"Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────────
# MODULE 3 – ANN + CROW SEARCH GA
# ─────────────────────────────────────────────
def run_crow_ga():
    global mse_crow
    if X_train is None:
        log("⚠  Please preprocess the dataset first."); return

    clear_log()
    log("=== ANN with Crow Search GA ===\n")
    n_total = X_train.shape[1]
    log(f"Total features in dataset before applying Crow Search GA : {n_total}")

    # Feature selection
    sel_idx = select_features_ga(X_train, y_train)
    log(f"Total features found after applying Crow Search GA       : {len(sel_idx)}")

    Xtr = X_train[:, sel_idx]
    Xte = X_test[:,  sel_idx]

    # Build ANN
    ann = SimpleANN(len(sel_idx), n_hidden=10, epochs=200, lr=0.01)

    # Optimise weights with Crow Search
    ann, best_fit = crow_search(ann, Xtr, y_train, n_crows=15, iters=40)

    # Fine-tune with back-prop
    ann.fit(Xtr, y_train)

    preds = ann.predict(Xte)
    mse_crow = mean_squared_error(y_test, preds)
    log(f"ANN with Crow Search MSE : {mse_crow:.4f}\n")

    # Show tabular results
    log(f"{'Algorithm':<30} {'Test Water Level':>18} {'Predicted Water Level':>22}")
    log("-" * 72)
    for i in range(len(y_test)):
        alg  = "ANN with Crow Search GA"
        tv   = float(y_test[i])
        pv   = float(preds[i])
        log(f"{alg:<30} {tv:>18.4f} {pv:>22.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_test,  label="Available Test Data Water Level", color="red",   linewidth=1.5)
    ax.plot(preds,   label="Predicted Water Level",           color="green", linewidth=1.5, linestyle="--")
    ax.set_title("ANN with Crow Search Water Level Prediction")
    ax.set_xlabel("Test Data Values")
    ax.set_ylabel("Water Level Prediction")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MODULE 4 – ANN + GREY WOLF GA
# ─────────────────────────────────────────────
def run_wolf_ga():
    global mse_wolf
    if X_train is None:
        log("⚠  Please preprocess the dataset first."); return

    clear_log()
    log("=== ANN with Grey Wolf GA ===\n")
    n_total = X_train.shape[1]
    log(f"Total features in dataset before applying Grey Wolf GA : {n_total}")

    # Feature selection
    sel_idx = select_features_ga(X_train, y_train)
    log(f"Total features found after applying Grey Wolf GA      : {len(sel_idx)}")

    Xtr = X_train[:, sel_idx]
    Xte = X_test[:,  sel_idx]

    # Build ANN
    ann = SimpleANN(len(sel_idx), n_hidden=10, epochs=200, lr=0.01)

    # Optimise weights with Grey Wolf
    ann, best_fit = grey_wolf(ann, Xtr, y_train, n_wolves=15, iters=40)

    # Fine-tune with back-prop
    ann.fit(Xtr, y_train)

    preds    = ann.predict(Xte)
    mse_wolf = mean_squared_error(y_test, preds)
    log(f"ANN with Grey Wolf MSE : {mse_wolf:.4f}\n")

    # Tabular results
    log(f"{'Algorithm':<30} {'Test Water Level':>18} {'Predicted Water Level':>22}")
    log("-" * 72)
    for i in range(len(y_test)):
        alg = "ANN with Grey Wolf GA"
        tv  = float(y_test[i])
        pv  = float(preds[i])
        log(f"{alg:<30} {tv:>18.4f} {pv:>22.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_test, label="Available Test Data Water Level", color="red",   linewidth=1.5)
    ax.plot(preds,  label="Predicted Water Level",           color="green", linewidth=1.5, linestyle="--")
    ax.set_title("ANN with Gray Wolf GA Water Level Prediction")
    ax.set_xlabel("Test Data Values")
    ax.set_ylabel("Water Level Prediction")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MODULE 5 – MSE COMPARISON GRAPH
# ─────────────────────────────────────────────
def show_mse_graph():
    if mse_crow is None or mse_wolf is None:
        log("⚠  Please run both algorithms first."); return

    clear_log()
    log(f"ANN with Crow Search GA  MSE : {mse_crow:.6f}")
    log(f"ANN with Grey Wolf GA    MSE : {mse_wolf:.6f}")
    winner = "Grey Wolf GA" if mse_wolf < mse_crow else "Crow Search GA"
    log(f"\nBetter algorithm (lower MSE) : {winner}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["ANN with Crow Search GA", "ANN with Gray Wolf GA"],
        [mse_crow, mse_wolf],
        color=["#1f77b4", "#4fa8e0"], width=0.4
    )
    ax.set_title("MSE ERROR")
    ax.set_xlabel("Algorithm Names")
    ax.set_ylabel("MSE")
    for bar, val in zip(bars, [mse_crow, mse_wolf]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("Groundwater Level Prediction Using Hybrid ANN with Genetic Algorithm")
root.geometry("900x640")
root.configure(bg="#1a1a2e")

# ── Title ────────────────────────────────────
tk.Label(
    root,
    text="Groundwater Level Prediction Using\nHybrid Artificial Neural Network with Genetic Algorithm",
    font=("Arial", 14, "bold"),
    fg="white", bg="#16213e",
    pady=10
).pack(fill="x")

# ── File label ───────────────────────────────
file_lbl = tk.Label(root, text="No file selected", font=("Arial", 9),
                    fg="#a0c4ff", bg="#1a1a2e", anchor="w", padx=10)
file_lbl.pack(fill="x")

# ── Button row 1 ─────────────────────────────
btn_frame1 = tk.Frame(root, bg="#1a1a2e", pady=6)
btn_frame1.pack(fill="x", padx=10)

btn_style = dict(font=("Arial", 10, "bold"), bg="#0f3460", fg="white",
                 relief="flat", cursor="hand2", padx=10, pady=5)

tk.Button(btn_frame1, text="Upload Ground Water Level Dataset",
          command=upload_dataset, **btn_style).pack(side="left", padx=4)

# ── Button row 2 ─────────────────────────────
btn_frame2 = tk.Frame(root, bg="#1a1a2e", pady=2)
btn_frame2.pack(fill="x", padx=10)

tk.Button(btn_frame2, text="Preprocess Dataset",
          command=preprocess_dataset, **btn_style).pack(side="left", padx=4)
tk.Button(btn_frame2, text="Run ANN with Crow Search GA",
          command=run_crow_ga, **btn_style).pack(side="left", padx=4)
tk.Button(btn_frame2, text="Run ANN with Grey Wolf GA",
          command=run_wolf_ga, **btn_style).pack(side="left", padx=4)

# ── Button row 3 ─────────────────────────────
btn_frame3 = tk.Frame(root, bg="#1a1a2e", pady=2)
btn_frame3.pack(fill="x", padx=10)

tk.Button(btn_frame3, text="MSE Comparison Graph",
          command=show_mse_graph, **btn_style).pack(side="left", padx=4)
tk.Button(btn_frame3, text="Exit",
          command=root.quit,
          font=("Arial", 10, "bold"), bg="#7a0000", fg="white",
          relief="flat", cursor="hand2", padx=10, pady=5
          ).pack(side="left", padx=4)

# ── Output box ───────────────────────────────
tk.Label(root, text="Output Console", font=("Arial", 10, "bold"),
         fg="#a0c4ff", bg="#1a1a2e").pack(anchor="w", padx=12, pady=(8,0))

output_box = scrolledtext.ScrolledText(
    root, font=("Courier", 9), bg="#0d0d1a", fg="#e0e0e0",
    state="disabled", height=26, relief="flat", bd=0
)
output_box.pack(fill="both", expand=True, padx=10, pady=(2, 10))

root.mainloop()
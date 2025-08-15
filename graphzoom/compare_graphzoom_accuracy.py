import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATASET = "cora"
LABELS_PATH = f"dataset/{DATASET}/{DATASET}-labels.npy"
EMB_MP = f"embed_results/{DATASET}_embedding_mpaware.npy"
EMB_NAIVE = f"embed_results/{DATASET}_embedding_naive.npy"

# Load labels
Y = np.load(LABELS_PATH)

def evaluate(X, name):
    split = len(Y) // 2  # 50% train / 50% test
    clf = LogisticRegression(max_iter=500)
    clf.fit(X[:split], Y[:split])
    pred = clf.predict(X[split:])
    acc = accuracy_score(Y[split:], pred)
    print(f"{name} accuracy: {acc:.4f}")

# Load embeddings
X_mpaware = np.load(EMB_MP).reshape(len(Y), -1)
X_naive = np.load(EMB_NAIVE).reshape(len(Y), -1)

# Evaluate
evaluate(X_naive, "Naive")
evaluate(X_mpaware, "MP-aware")

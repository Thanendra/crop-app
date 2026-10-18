import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

GREEN, WHEAT, GREY = "#14382a", "#e6b64a", "#b9c4bd"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 15,
    "font.size": 11, "savefig.facecolor": "white",
})
NICE = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
        "temperature": "Temperature", "humidity": "Humidity",
        "ph": "Soil pH", "rainfall": "Rainfall"}

df = pd.read_csv("Crop_recommendation.csv")
X = df.drop(columns=["label"])
y = df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Naive Bayes": GaussianNB(),
}

os.makedirs("static/charts", exist_ok=True)
scores = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    scores[name] = round(accuracy_score(y_te, m.predict(X_te)) * 100, 2)
    print(name, scores[name])

# 1) Model comparison
names = sorted(scores, key=scores.get)
vals = [scores[n] for n in names]
colors = [GREEN if n == "Random Forest" else GREY for n in names]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(names, vals, color=colors, height=0.6)
ax.set_xlim(90, 100.6)
ax.set_xlabel("Test accuracy (%)")
ax.set_title("Model comparison", loc="left")
for b, v in zip(bars, vals):
    ax.text(v + 0.08, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
            va="center", fontweight="bold")
fig.tight_layout()
fig.savefig("static/charts/comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 2) Confusion matrix (Random Forest)
rf = models["Random Forest"]
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay.from_estimator(
    rf, X_te, y_te, ax=ax, cmap="Greens", colorbar=False,
    xticks_rotation=90, values_format="d")
for t in disp.text_.ravel():
    if t.get_text() == "0":
        t.set_text("")
    t.set_fontsize(9)
ax.set_title("Confusion matrix - Random Forest", loc="left")
ax.set_xlabel("Predicted crop")
ax.set_ylabel("Actual crop")
fig.tight_layout()
fig.savefig("static/charts/confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 3) Feature importance
imp = (pd.Series(rf.feature_importances_, index=X.columns) * 100).sort_values()
colors = [GREEN if v == imp.max() else WHEAT for v in imp.values]
labels = [NICE.get(i, i) for i in imp.index]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(labels, imp.values, color=colors, height=0.6)
ax.set_xlabel("Importance (%)")
ax.set_title("Which input matters most?", loc="left")
for b, v in zip(bars, imp.values):
    ax.text(v + 0.2, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontweight="bold")
ax.set_xlim(0, imp.max() + 4)
fig.tight_layout()
fig.savefig("static/charts/feature_importance.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("Done. Charts saved in static/charts/")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# 2. Select input features and target
X = data[
    ["N", "P", "K", "temperature",
     "humidity", "ph", "rainfall"]
]
y = data["label"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Check model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model
joblib.dump(model, "crop_model.pkl")

print("Model trained successfully!")
print("Saved as crop_model.pkl")
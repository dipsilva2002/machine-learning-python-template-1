import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

results = []
for n_estimators in [50, 100, 200]:
    for max_depth in [3, 5, 10, None]:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append({"n_estimators": n_estimators, "max_depth": max_depth, "accuracy": accuracy})

results_df = pd.DataFrame(results)
print(results_df)

best_idx = results_df["accuracy"].idxmax()
best = results_df.loc[best_idx]

best_n = int(best["n_estimators"])
best_depth = None if pd.isna(best["max_depth"]) else int(best["max_depth"])

final_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/random_forest_model.pkl")
print("Modelo guardado con:", {"n_estimators": best_n, "max_depth": best_depth, "accuracy": float(best["accuracy"])})


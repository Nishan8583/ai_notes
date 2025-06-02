# Just a copy paste from ChatGPT
import pandas as pd
import joblib

# Load model and vectorizer
clf = joblib.load("isolation_forest_model.pkl")
vec = joblib.load("tfidf_vectorizer.pkl")

# Load new logs
new_df = pd.read_csv("unseen_logs.csv")
X_new = vec.transform(new_df["CommandLine"])

# Predict anomalies
preds = clf.predict(X_new)
# -1 = anomaly (suspicious), 1 = normal
new_df["anomaly"] = preds
anomalies = new_df[new_df["anomaly"] == -1]

# Save suspicious logs
anomalies.to_csv("suspicious_logs.csv", index=False)

print(f"🔍 Found {len(anomalies)} suspicious command lines")

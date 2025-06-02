# Just a copy paste from ChatGPT
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import joblib

# Load your benign logs
df = pd.read_csv("benign_logs.csv")  # must have a 'CommandLine' column
df.dropna(subset=["CommandLine"], inplace=True)

# TF-IDF vectorization
'''
Turns your CommandLine strings into numerical vectors so a model can work with them. It breaks each string into tokens (words/arguments), computes term frequency (TF), adjusts with inverse document frequency (IDF), and returns a matrix.
| Parameter           | What It Does                                                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `max_features=1000` | Limits the vocabulary to the **top 1000 most frequent tokens** across all logs. This keeps the model fast and avoids overfitting. |
| `lowercase=True`    | Converts all command line text to **lowercase**, so `"PowerShell"` and `"powershell"` are treated the same.                       |

'''
vec = TfidfVectorizer(max_features=1000, lowercase=True)
X = vec.fit_transform(df["CommandLine"])

# Train Isolation Forest
# Fits an anomaly detection model that builds random trees to isolate points. If a point is isolated very quickly (i.e., it lies in a sparse region), it's likely anomalous.
'''
| Parameter            | What It Does                                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `contamination=0.01` | Specifies that **\~1%** of the training data is expected to be anomalous (outliers). This affects the threshold used to decide what's normal vs. abnormal. |
| `random_state=42`    | Sets the **random seed** so that results are reproducible. (Change this to get different trees.)                                                           |

'''
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)

# Save the model and vectorizer
joblib.dump(clf, "isolation_forest_model.pkl")
joblib.dump(vec, "tfidf_vectorizer.pkl")  # we need to save the vectorizer because we need to transform new logs the same way
'''
If not the following might happen:
"powershell" → index 12
"-nop"        → index 53
"-enc"        → index 78

"powershell" → index 4
"-nop"        → index 23
"-enc"        → index 67

Now the model would get completely mismatched input → predictions would be meaningless.The new model will get  

'''

print("✅ Model and vectorizer saved!")

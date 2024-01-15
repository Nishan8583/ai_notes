import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("credit_data.csv")
#print(df.head())
#print(df.describe())
#print(df.corr())
features=df[["income","age","loan"]]# x
target=df["default"] # y

# 30% test is for testing 70% for training
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=0.3)

model = LogisticRegression()
model.fit(features_train,target_train)

predictions=model.predict(features_test)
print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))
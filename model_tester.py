import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_df = pd.read_csv("data/iris_train.csv")

x = training_df.drop(columns=["species"])
y = training_df["species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)

model.fit(X_train,Y_train)

predict_test = model.predict(X_test)
baseline_accuracy = accuracy_score(Y_test, predict_test)
print("Baseline Accuracy: ", round(baseline_accuracy*100,2),"%")

drift_df = pd.read_csv("data/iris_drifted.csv")

x_drift = drift_df.drop(columns=["species"])
y_drift = drift_df["species"]

predicted_drift = model.predict(x_drift)
drifted_accuracy = accuracy_score(y_drift, predicted_drift)
print("Drift Accuracy: ", round(drifted_accuracy*100,2), "%")

print("Drop in Accuracy: ", round(((baseline_accuracy-drifted_accuracy)*100),2),"%")
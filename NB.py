import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("iris.csv")
print("Dataset loaded successfully.\n")

print("First 5 rows of the dataset:\n")
print(df.head())
print("\n")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("Data split into training and testing sets.\n")

model = GaussianNB()
model.fit(X_train, y_train)
print("Model trained successfully.\n")

y_pred = model.predict(X_test)

print("Predictions:\n", y_pred, "\n")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Naive Bayes classification completed successfully.")

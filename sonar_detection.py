import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- Data loading ---
sonar_data = pd.read_csv(r"sonar.csv", header=None)

x = sonar_data.drop(columns=60)
y = sonar_data[60]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1
)

# --- Feature scaling (KNN is distance-based, so scaling is important) ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Train KNN model ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled, y_train)

# --- Evaluate ---
train_acc = accuracy_score(y_train, knn_model.predict(x_train_scaled))
test_acc = accuracy_score(y_test, knn_model.predict(x_test_scaled))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, knn_model.predict(x_test_scaled)))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_model.predict(x_test_scaled)))

# --- Making a prediction on new/sample data ---
# Example: predicting on a single row from the test set (as a sanity check)
input_data = x_test.iloc[0].values.reshape(1, -1)
input_data_scaled = scaler.transform(input_data)
prediction = knn_model.predict(input_data_scaled)

print(f"\nSample prediction: {prediction[0]} (Actual: {y_test.iloc[0]})")
if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")


import pickle

# Save the trained model
with open('knn_model.sav', 'wb') as f:
    pickle.dump(knn_model, f)

# Save the scaler too — you MUST use the same scaler for any future predictions
with open('scaler.sav', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
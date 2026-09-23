import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Import multiple classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- Data loading (same as before) ---
sonar_data = pd.read_csv(r"sonar.csv", header=None)

x = sonar_data.drop(columns=60)
y = sonar_data[60]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1
)

# Some models (SVM, KNN, LogisticRegression) are sensitive to feature scale
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Define models to compare ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf'),
    "SVM (Linear)": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():
    # Use scaled data for distance/margin-based models, raw for tree-based ones
    if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        model.fit(x_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(x_train))
        test_acc = accuracy_score(y_test, model.predict(x_test))
    else:
        model.fit(x_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model.predict(x_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(x_test_scaled))

    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4)
    })

results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model based on test accuracy: {best_model_name}")
import numpy as np
import pandas as pd
import pickle
import time
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sonar: Rock vs Mine Prediction",
    page_icon="🌊",
    layout="wide"
)

# =========================================================
# SAMPLE DATA (used for quick demo predictions)
# =========================================================
SAMPLE_ROWS = {
    "Sample A (Rock)": [
        0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111,
        0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797,
        0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857,
        0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
        0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324,
        0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032
    ],
    "Sample B (Mine)": [
        0.0392, 0.0108, 0.0267, 0.0257, 0.0410, 0.0491, 0.1053, 0.1690, 0.2105, 0.2471,
        0.2680, 0.3049, 0.2863, 0.2294, 0.1165, 0.2469, 0.3830, 0.4956, 0.5031, 0.5061,
        0.5990, 0.6975, 0.8332, 0.9970, 1.0000, 0.9945, 0.9737, 0.9675, 0.9746, 0.9111,
        0.7237, 0.4884, 0.4491, 0.6042, 0.7048, 0.7943, 0.8785, 0.8556, 0.7515, 0.5904,
        0.4192, 0.3617, 0.4310, 0.4839, 0.4658, 0.3773, 0.2413, 0.1341, 0.0864, 0.0392,
        0.0193, 0.0157, 0.0180, 0.0100, 0.0117, 0.0112, 0.0100, 0.0088, 0.0056, 0.0070
    ],
}

MODEL_ZOO = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "Naive Bayes": GaussianNB(),
}

TREE_BASED = {"Decision Tree", "Random Forest", "Gradient Boosting"}

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    """Load sonar.csv from the working directory (no header)."""
    return pd.read_csv("sonar.csv", header=None)


@st.cache_resource
def train_all_models(_x_train, _y_train, _x_train_scaled):
    """Train every model in MODEL_ZOO and cache the fitted objects."""
    fitted = {}
    for name, model in MODEL_ZOO.items():
        if name in TREE_BASED:
            model.fit(_x_train, _y_train)
        else:
            model.fit(_x_train_scaled, _y_train)
        fitted[name] = model
    return fitted


def evaluate_models(fitted_models, x_train, x_test, y_train, y_test, x_train_s, x_test_s):
    rows = []
    for name, model in fitted_models.items():
        if name in TREE_BASED:
            train_acc = accuracy_score(y_train, model.predict(x_train))
            test_acc = accuracy_score(y_test, model.predict(x_test))
        else:
            train_acc = accuracy_score(y_train, model.predict(x_train_s))
            test_acc = accuracy_score(y_test, model.predict(x_test_s))
        rows.append({"Model": name, "Train Accuracy": round(train_acc, 4), "Test Accuracy": round(test_acc, 4)})
    return pd.DataFrame(rows).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)


def predict_with_model(name, model, scaler, row_df):
    if name in TREE_BASED:
        X = row_df
    else:
        X = scaler.transform(row_df)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    return pred, proba


# =========================================================
# SIDEBAR: DATA SOURCE + TRAIN/TEST SETTINGS
# =========================================================
st.sidebar.title("⚙️ Setup")

try:
    sonar_data = load_data()
    data_loaded = True
except FileNotFoundError:
    sonar_data = None
    data_loaded = False

test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.1, 0.05)
random_state = st.sidebar.number_input("Random state", value=1, step=1)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Data Exploration", "🤖 Model Comparison", "🔮 Make a Prediction"]
)

# =========================================================
# PREP: split + scale (only if data is loaded)
# =========================================================
if data_loaded:
    x = sonar_data.drop(columns=60)
    y = sonar_data[60]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=int(random_state)
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.title("🌊 Sonar Signal Classification: Rock vs Mine")
    st.markdown("""
    This app classifies sonar return signals as either a **Rock (R)** or a **Mine (M)**,
    based on the classic Sonar dataset (60 frequency-band energy features per sample, 208 total samples).

    **What this app demonstrates:**
    - End-to-end ML pipeline: load → split → scale → train → evaluate
    - Comparison across 8 classification algorithms
    - Live predictions using sample data, manual input, or an uploaded CSV
    """)

    col1, col2, col3 = st.columns(3)
    if data_loaded:
        col1.metric("Total Samples", len(sonar_data))
        col2.metric("Features", x.shape[1])
        col3.metric("Classes", sonar_data[60].nunique())
    else:
        st.warning(
            "No `sonar.csv` found in the working directory. Place it next to `app.py` "
            "to unlock Data Exploration and Model Comparison. "
            "You can still try predictions on the next tab using built-in samples if you "
            "have a pre-trained `knn_model.sav` / `scaler.sav`."
        )

# =========================================================
# PAGE: DATA EXPLORATION
# =========================================================
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    if not data_loaded:
        st.error("Place sonar.csv next to app.py to explore the data.")
    else:
        st.subheader("Raw Data")
        st.dataframe(sonar_data.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots()
            sonar_data[60].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
            ax.set_xlabel("Class (R = Rock, M = Mine)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.subheader("Feature Value Distribution (mean per sample)")
            fig2, ax2 = plt.subplots()
            x.mean(axis=1).plot(kind="hist", bins=20, ax=ax2, color="#55A868")
            ax2.set_xlabel("Mean signal strength")
            st.pyplot(fig2)

        st.subheader("Summary Statistics")
        st.dataframe(x.describe())

# =========================================================
# PAGE: MODEL COMPARISON
# =========================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    if not data_loaded:
        st.error("Place sonar.csv next to app.py to train and compare models.")
    else:
        with st.spinner("Training all models..."):
            fitted_models = train_all_models(x_train, y_train, x_train_scaled)
            results_df = evaluate_models(
                fitted_models, x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled
            )

        st.subheader("Results Table")
        st.dataframe(
            results_df.style.highlight_max(subset=["Test Accuracy"], color="lightgreen"),
            use_container_width=True
        )

        best_row = results_df.iloc[0]
        st.success(f"🏆 Best model on test accuracy: **{best_row['Model']}** ({best_row['Test Accuracy']:.2%})")

        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(9, 4))
        x_pos = np.arange(len(results_df))
        width = 0.35
        ax.bar(x_pos - width/2, results_df["Train Accuracy"], width, label="Train")
        ax.bar(x_pos + width/2, results_df["Test Accuracy"], width, label="Test")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Inspect a Model")
        chosen = st.selectbox("Select a model to inspect", results_df["Model"])
        model = fitted_models[chosen]
        X_eval = x_test if chosen in TREE_BASED else x_test_scaled
        preds = model.predict(X_eval)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, preds, labels=model.classes_)
            fig_cm, ax_cm = plt.subplots()
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks(range(len(model.classes_)))
            ax_cm.set_yticks(range(len(model.classes_)))
            ax_cm.set_xticklabels(model.classes_)
            ax_cm.set_yticklabels(model.classes_)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
            st.pyplot(fig_cm)

        with col2:
            st.write("**Classification Report**")
            report = classification_report(y_test, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3))

        # Let the user download the trained model + scaler
        st.subheader("Export This Model")
        model_bytes = pickle.dumps(model)
        scaler_bytes = pickle.dumps(scaler)
        col1, col2 = st.columns(2)
        col1.download_button(f"⬇️ Download {chosen} (.sav)", data=model_bytes, file_name=f"{chosen.replace(' ', '_')}_model.sav")
        col2.download_button("⬇️ Download scaler (.sav)", data=scaler_bytes, file_name="scaler.sav")

# =========================================================
# PAGE: MAKE A PREDICTION
# =========================================================
elif page == "🔮 Make a Prediction":
    st.title("🔮 Make a Prediction")

    # Determine which models/scaler are available: freshly trained (if data loaded)
    # or loaded from saved .sav files as a fallback.
    fitted_models = None
    active_scaler = None

    if data_loaded:
        with st.spinner("Preparing models..."):
            fitted_models = train_all_models(x_train, y_train, x_train_scaled)
            active_scaler = scaler
    else:
        try:
            fitted_models = {"KNN (loaded)": pickle.load(open("knn_model.sav", "rb"))}
            active_scaler = pickle.load(open("scaler.sav", "rb"))
            st.info("Using pre-trained `knn_model.sav` since no sonar.csv is loaded.")
        except FileNotFoundError:
            st.error(
                "No data and no saved model found. Place sonar.csv, or "
                "knn_model.sav / scaler.sav, next to app.py."
            )

    if fitted_models:
        model_name = st.selectbox("Choose a model", list(fitted_models.keys()))
        model = fitted_models[model_name]

        input_method = st.radio(
            "Input method",
            ["Use a built-in sample", "Manual entry"],
            horizontal=True
        )

        input_df = None

        if input_method == "Use a built-in sample":
            choice = st.selectbox("Pick a sample", list(SAMPLE_ROWS.keys()))
            input_df = pd.DataFrame([SAMPLE_ROWS[choice]])
            st.dataframe(input_df)

        elif input_method == "Manual entry":
            st.caption("Adjust a few key features; the rest default to the dataset mean (or 0.2 if no data loaded).")
            defaults = x.mean().values if data_loaded else np.full(60, 0.2)
            cols = st.columns(6)
            values = list(defaults)
            for i in range(10):  # expose first 10 features as sliders for a quick demo
                with cols[i % 6]:
                    values[i] = st.slider(f"F{i+1}", 0.0, 1.0, float(defaults[i]), 0.01)
            input_df = pd.DataFrame([values])

        if st.button("Predict", type="primary"):
            if input_df is None:
                st.warning("Provide input data first.")
            elif input_df.shape[1] != 60:
                st.error(f"Expected 60 features, got {input_df.shape[1]}.")
            else:
                with st.spinner("Predicting..."):
                    time.sleep(0.3)
                    for i in range(len(input_df)):
                        row = input_df.iloc[[i]]
                        pred, proba = predict_with_model(model_name, model, active_scaler, row)
                        label = "🪨 Rock" if pred == "R" else "💣 Mine"
                        st.subheader(f"Row {i+1}: {label}")
                        if proba:
                            st.bar_chart(pd.Series(proba))

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · Sonar dataset")
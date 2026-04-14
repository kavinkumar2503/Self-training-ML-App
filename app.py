# ==============================
# FINAL SELF-TRAINING WEB APP (NO ROW LOSS VERSION)
# ==============================
# Run:
# pip install streamlit scikit-learn pandas numpy matplotlib
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Self-Training ML App", layout="wide")

st.title("🤖 Self-Training Pipeline for Partially Labeled Data")
st.write("Upload dataset → Train → Auto-label → Improve model")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    target_column = st.selectbox("Select Target Column", data.columns)

    if st.button("Start Training"):
        df = data.copy().reset_index(drop=True)
        
        # ==============================
        # INITIAL MISSING VALUE ANALYSIS
        # ==============================
        initial_missing = df[target_column].isna().sum()
        st.subheader("🔍 Missing Values Analysis")
        st.write(f"Initial missing values in target: {initial_missing}")
        original_count = len(df)

        # Split
        labeled = df[df[target_column].notna()].copy().reset_index(drop=True)
        unlabeled = df[df[target_column].isna()].copy().reset_index(drop=True)

        st.write(f"Labeled samples: {len(labeled)}")
        st.write(f"Unlabeled samples: {len(unlabeled)}")

        if len(unlabeled) == 0:
            st.warning("⚠️ No unlabeled data found. Please remove some target values.")

        # Features & target
        X_labeled = labeled.drop(columns=[target_column])
        y_labeled = labeled[target_column]
        X_unlabeled = unlabeled.drop(columns=[target_column])

        # Keep numeric only
        X_labeled = X_labeled.select_dtypes(include=[np.number])
        X_unlabeled = X_unlabeled.select_dtypes(include=[np.number])

        # Fill missing
        X_labeled = X_labeled.fillna(X_labeled.mean())
        X_unlabeled = X_unlabeled.fillna(X_unlabeled.mean())

        # Encode labels
        le = LabelEncoder()
        y_labeled_encoded = le.fit_transform(y_labeled)

        # Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_labeled, y_labeled_encoded)

        accuracies = []

        # Self-training loop
        for i in range(5):
            if len(X_unlabeled) == 0:
                break

            probs = model.predict_proba(X_unlabeled)
            confidence = np.max(probs, axis=1)
            preds = model.predict(X_unlabeled)

            threshold = 0.9
            high_conf_idx = np.where(confidence > threshold)[0]

            if len(high_conf_idx) == 0:
                break

            # Add pseudo-labeled data
            X_new = X_unlabeled.iloc[high_conf_idx]
            y_new = preds[high_conf_idx]

            X_labeled = pd.concat([X_labeled, X_new], ignore_index=True)
            y_labeled_encoded = np.concatenate([y_labeled_encoded, y_new])

            # SAFE removal (NO ROW LOSS)
            mask = np.ones(len(X_unlabeled), dtype=bool)
            mask[high_conf_idx] = False
            X_unlabeled = X_unlabeled.iloc[mask].reset_index(drop=True)

            # Retrain
            model.fit(X_labeled, y_labeled_encoded)

            # Evaluate
            y_pred = model.predict(X_labeled)
            acc = accuracy_score(y_labeled_encoded, y_pred)
            accuracies.append(acc)

        # ==============================
        # MODEL PERFORMANCE METRICS
        # ==============================
        st.subheader("📊 Model Performance")

        from sklearn.metrics import precision_score, recall_score, f1_score

        y_pred_final = model.predict(X_labeled)

        acc_final = accuracy_score(y_labeled_encoded, y_pred_final)
        precision = precision_score(y_labeled_encoded, y_pred_final, average='weighted', zero_division=0)
        recall = recall_score(y_labeled_encoded, y_pred_final, average='weighted', zero_division=0)
        f1 = f1_score(y_labeled_encoded, y_pred_final, average='weighted', zero_division=0)

        st.write(f"Accuracy: {acc_final:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Results
        st.subheader("Final Model Accuracy")
        if accuracies:
            st.success(f"Accuracy: {accuracies[-1]:.2f}")
        else:
            st.warning("No self-training iterations completed")

        # Graph
        if accuracies:
            st.subheader("Accuracy Improvement Graph")
            fig, ax = plt.subplots()
            ax.plot(accuracies, marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")
            st.pyplot(fig)

        # ==============================
        # HANDLE REMAINING UNLABELED DATA (PREVENT ROW LOSS)
        # ==============================
        if len(X_unlabeled) > 0:
            # Predict remaining unlabeled data
            remaining_preds = model.predict(X_unlabeled)
            X_labeled = pd.concat([X_labeled, X_unlabeled], ignore_index=True)
            y_labeled_encoded = np.concatenate([y_labeled_encoded, remaining_preds])

        # Convert labels back
        final_labels = le.inverse_transform(y_labeled_encoded.astype(int))

        final_df = pd.concat([
            pd.DataFrame(X_labeled, columns=X_labeled.columns),
            pd.Series(final_labels, name=target_column)
        ], axis=1)

        # ==============================
        # FINAL MISSING VALUE ANALYSIS
        # ==============================
        final_missing = pd.Series(final_labels).isna().sum()
        st.write(f"Final missing values in target: {final_missing}")

        improvement = initial_missing - final_missing
        st.write(f"Missing values filled by model: {improvement}")

        # VALIDATION (ROW CHECK)
        st.write(f"Original rows: {original_count}")
        st.write(f"Final rows: {len(final_df)}")

        if original_count == len(final_df):
            st.success("✅ No data loss. All rows preserved.")
        else:
            st.error("❌ Row mismatch detected!")

        # Download
        st.subheader("Download Processed Data")
        st.download_button(
            "Download CSV",
            final_df.to_csv(index=False),
            "output.csv"
        )

else:
    st.info("Please upload a dataset to begin.")


# antibody_yield_explainer_app.py

"""
A simplified Streamlit app to predict monoclonal antibody (mAb) yield
from just the Heavy and Light chain sequences.

This app uses ESM2 embeddings + default metadata, runs a trained model,
and returns:
- Predicted yield
- Metadata assumptions
- Top contributing features

Files required:
- yield_model.pkl
- meta_columns.pkl

Run using:
    streamlit run antibody_yield_explainer_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import esm
import joblib

# Load model and metadata structure
regressor = joblib.load("yield_model.pkl")
meta_columns = joblib.load("meta_columns.pkl")

# Load ESM2 model
@st.cache_resource
def load_esm():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.cpu()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, batch_converter

model, batch_converter = load_esm()

def embed_sequence(seq):
    seq = seq.replace("*", "").replace("\n", "").strip().upper()
    seq = ''.join([aa for aa in seq if aa in "ACDEFGHIKLMNPQRSTVWY"])
    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
    token_embeddings = results["representations"][6]
    return token_embeddings[0, 1:len(seq)+1].mean(0).numpy()

def predict_yield(hv_seq, lv_seq):
    hv_emb = embed_sequence(hv_seq)
    lv_emb = embed_sequence(lv_seq)
    X_seq = np.hstack([hv_emb, lv_emb])

    # Default metadata values
    metadata = {
        "Codon Optimized": 1,
        "Signal Peptide": "osteonectin",
        "Vector": "pTT5",
        "Expression System": "HEK293",
        "Temperature": 37
    }
    df_input = pd.DataFrame([metadata])
    df_input.columns = df_input.columns.str.strip()
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=meta_columns, fill_value=0)

    X_final = np.hstack([X_seq, df_encoded.values[0]])
    prediction = regressor.predict([X_final])[0]
    return prediction, metadata, df_encoded.columns[df_encoded.values[0] == 1].tolist()

# --- Streamlit Interface ---
st.title("🧪 mAb Yield Prediction from Sequence Only")

st.markdown("Paste your Heavy and Light chain sequences below. We'll use default metadata and return the predicted yield + factors considered.")

hv_seq = st.text_area("Heavy Chain Sequence (HV)", height=180)
lv_seq = st.text_area("Light Chain Sequence (LV)", height=180)

if st.button("Predict Yield"):
    if not hv_seq or not lv_seq:
        st.warning("Please enter both heavy and light chain sequences.")
    else:
        with st.spinner("Analyzing sequences and predicting yield..."):
            pred_yield, used_metadata, contributing_features = predict_yield(hv_seq, lv_seq)

        st.success(f"🧪 Predicted Yield: {pred_yield:.2f} mg per 100 mL")

        st.markdown("---")
        st.subheader("📋 Metadata Assumptions")
        for k, v in used_metadata.items():
            st.write(f"**{k}:** {v}")

        st.subheader("🔍 Features Considered")
        st.write("These are the key metadata features that contributed to the prediction:")
        st.code(", ".join(contributing_features), language='markdown')

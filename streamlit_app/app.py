import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from fashion_clip.fashion_clip import FashionCLIP

EMBED_PATH = "features/fashionclip_features.npz"
META_PATH = "data/metadata.csv"
IMG_DIR = "data/images"
TOPK = 6
MIN_SIM = 0.2

@st.cache_resource
def load_model():
    model = FashionCLIP("fashion-clip")
    return model

@st.cache_data
def load_data():
    arr = np.load(EMBED_PATH)
    meta = pd.read_csv(META_PATH)
    return arr["image_embeddings"], arr["image_paths"], arr["texts"], meta

def get_top_matches(query, model, img_embs, img_paths):
    query_emb = model.encode_text([query], batch_size=1)[0]
    img_embs_tensor = torch.tensor(img_embs)
    query_emb_tensor = torch.tensor(query_emb)
    sims = torch.nn.functional.cosine_similarity(
        img_embs_tensor, query_emb_tensor.unsqueeze(0), dim=1
    ).numpy()
    topk_idx = np.argsort(sims)[::-1][:TOPK]
    top_img_paths = [os.path.join(IMG_DIR, os.path.basename(img_paths[i])) for i in topk_idx]
    top_scores = [sims[i] for i in topk_idx]
    return top_img_paths, top_scores

def main():
    st.title("From Words to Wardrobe: Fashion Search")

    # Loading block
    if "model_loaded" not in st.session_state:
        with st.spinner("Hold on before you enter that search query; setting up things for you..."):
            model = load_model()
            img_embs, img_paths, texts, meta = load_data()
            st.session_state["model"] = model
            st.session_state["img_embs"] = img_embs
            st.session_state["img_paths"] = img_paths
            st.session_state["texts"] = texts
            st.session_state["meta"] = meta
            st.session_state["model_loaded"] = True
    else:
        model = st.session_state["model"]
        img_embs = st.session_state["img_embs"]
        img_paths = st.session_state["img_paths"]
        texts = st.session_state["texts"]
        meta = st.session_state["meta"]

    # Input field and button
    query = st.text_input("Search for a fashion product by description and see the top 6 results.", value="yellow floral dress")
    if st.button("Search"):
        with st.spinner("Wait while we search through hundreds of products to get you the best match..."):
            top_img_paths, top_scores = get_top_matches(query, model, img_embs, img_paths)
            avg_score = np.mean(top_scores)
        st.subheader("Top 6 Results")
        if avg_score <= MIN_SIM:
            st.info('Not a perfect match, but here’s the closest we could find!') 
        # 2x3 grid
        cols = st.columns(3)
        for i, img_path in enumerate(top_img_paths):
            with cols[i % 3]:
                st.image(img_path, use_container_width=True)

if __name__ == "__main__":
    main()
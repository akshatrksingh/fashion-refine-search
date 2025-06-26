import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

import torch
from fashion_clip.fashion_clip import FashionCLIP

EMBED_PATH = "features/fashionclip_features.npz"
META_PATH = "data/metadata.csv"
IMG_DIR = "data/images"
OUT_DIR = "sample_run"
TOPK = 6
MIN_SIM = 0.2

def load_embeddings():
    print("Loading embeddings...")
    arr = np.load(EMBED_PATH)
    return arr["image_embeddings"], arr["text_embeddings"], arr["image_paths"], arr["texts"]

def plot_grid(image_paths, scores, query, out_path):
    plt.figure(figsize=(12, 8))
    for idx, img_path in enumerate(image_paths):
        plt.subplot(2, 3, idx + 1)
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
        except Exception:
            plt.text(0.5, 0.5, "Missing\nImage", ha="center", va="center", fontsize=18)
            plt.axis('off')
        plt.title(f"Score: {scores[idx]:.2f}", fontsize=10)
        plt.axis('off')
    plt.suptitle(f"Top Matches for Query: \"{query}\"", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve.py \"your text query here\"")
        sys.exit(1)
    query = sys.argv[1].strip().lower()
    if len(query) < 3:
        print("Query too short. Showing fallback results.")
    
    print(f"Loading embeddings and metadata...")
    img_embs, txt_embs, img_paths, texts = load_embeddings()
    meta = pd.read_csv(META_PATH)
    
    print("Loading FashionCLIP model...")
    model = FashionCLIP("fashion-clip")
    DEVICE = "cpu" 

    print("Embedding query text...")
    query_emb = model.encode_text([query], batch_size=1)[0]

    print("Computing similarities...")
    img_embs_tensor = torch.tensor(img_embs)
    query_emb_tensor = torch.tensor(query_emb)
    sims = torch.nn.functional.cosine_similarity(
        img_embs_tensor, query_emb_tensor.unsqueeze(0), dim=1
    ).numpy()

    # Get top-k
    topk_idx = np.argsort(sims)[::-1][:TOPK]
    top_img_paths = [os.path.join(IMG_DIR, os.path.basename(img_paths[i])) for i in topk_idx]
    top_scores = [sims[i] for i in topk_idx]

    print("\n=== Top 6 Results ===")
    for i, (img_path, score) in enumerate(zip(top_img_paths, top_scores)):
        print(f"{i+1}. {img_path} | Similarity: {score:.3f}")

    avg_score = np.mean(top_scores)
    if avg_score <= MIN_SIM:
        print('\nNot a perfect match, but here’s the closest we could find! \U0001F441')

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query.replace(' ', '_')[:30]}.png"
    )
    out_path = os.path.join(OUT_DIR, out_name)
    plot_grid(top_img_paths, top_scores, query, out_path)
    print(f"\nSaved plot to: {out_path}")

if __name__ == "__main__":
    main()
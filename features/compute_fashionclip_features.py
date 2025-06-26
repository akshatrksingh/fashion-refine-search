import os
import pandas as pd
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
from tqdm import tqdm

DEVICE = "cpu"
print("Using device:", DEVICE)
model = FashionCLIP("fashion-clip")

df = pd.read_csv("data/metadata.csv")
image_paths = df["image_path"].tolist()
texts = df["text"].tolist()

BATCH_SIZE = 32

all_img_embeds = []
all_txt_embeds = []
kept_paths = []
kept_texts = []

print("Computing embeddings in batches...")

for start in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[start:start+BATCH_SIZE]
    batch_texts = texts[start:start+BATCH_SIZE]

    abs_paths = [os.path.abspath(os.path.join("data", p)) for p in batch_paths]
    valid_idx = [i for i, p in enumerate(abs_paths) if os.path.exists(p)]
    if not valid_idx:
        continue

    valid_paths = [abs_paths[i] for i in valid_idx]
    valid_texts = [batch_texts[i] for i in valid_idx]

    try:
        img_embeds = model.encode_images(valid_paths, batch_size=len(valid_paths))
        txt_embeds = model.encode_text(valid_texts, batch_size=len(valid_texts))
        all_img_embeds.extend(img_embeds)
        all_txt_embeds.extend(txt_embeds)
        kept_paths.extend([batch_paths[i] for i in valid_idx])
        kept_texts.extend(valid_texts)
    except Exception as e:
        print(f"Error on batch {start}: {e}")

os.makedirs("features", exist_ok=True)
np.savez_compressed(
    "features/fashionclip_features.npz",
    image_embeddings=np.array(all_img_embeds),
    text_embeddings=np.array(all_txt_embeds),
    image_paths=np.array(kept_paths),
    texts=np.array(kept_texts),
)

print(f"Saved {len(kept_paths)} pairs to features/fashionclip_features.npz")
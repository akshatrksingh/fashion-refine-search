import numpy as np

data = np.load("features/fashionclip_features.npz")

print("Image embeddings shape:", data["image_embeddings"].shape)
print("Text embeddings shape:", data["text_embeddings"].shape)

print("Image embeddings all zero?", np.all(data["image_embeddings"] == 0))
print("Text embeddings all zero?", np.all(data["text_embeddings"] == 0))
print("Any NaNs in image embeddings?", np.isnan(data["image_embeddings"]).any())
print("Any NaNs in text embeddings?", np.isnan(data["text_embeddings"]).any())
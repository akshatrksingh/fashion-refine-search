import os
import requests
import pandas as pd

# 1. Create output dir
os.makedirs("images", exist_ok=True)

# 2. API config
total_to_fetch = 5000  
batch_size = 100
base_url = "https://datasets-server.huggingface.co/rows?dataset=Marqo%2Ffashion200k&config=default&split=data"

# 3. Track seen product IDs to avoid duplicates
seen_products = set()
records = []

print(f"Fetching up to {total_to_fetch} unique products...")

for offset in range(0, total_to_fetch, batch_size): 
    if len(seen_products) >= total_to_fetch:
        break

    url = f"{base_url}&offset={offset}&length={batch_size}"
    print(f"Fetching rows {offset} to {offset + batch_size}...")

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        rows = resp.json()["rows"]
    except Exception as e:
        print("Error fetching metadata:", e)
        print("Response content:", resp.text)
        break

    for row in rows:
        row_data = row["row"]
        item_id = row_data["item_ID"]
        product_id = "_".join(item_id.split("_")[:-1])

        if product_id in seen_products:
            continue
        seen_products.add(product_id)

        image_url = row_data["image"]["src"]
        text = row_data["text"]
        fname = f"{item_id}.jpg"
        local_path = f"images/{fname}"

        try:
            img_data = requests.get(image_url, timeout=10).content
            with open(local_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")
            continue

        records.append({
            "image_path": local_path,
            "text": text,
            "item_ID": item_id,
            "product_id": product_id,
        })

# 4. Save metadata
df = pd.DataFrame(records)
df.to_csv("metadata.csv", index=False)
print(f"Saved {len(df)} unique products to metadata.csv")
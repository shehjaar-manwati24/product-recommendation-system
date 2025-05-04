import json
import csv
import os

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# File paths (you already decompressed them)
REVIEWS_PATH = 'Electronics.json'
META_PATH = ' meta_Electronics.json'

# Output CSV paths
INTERACTIONS_CSV = 'data/interactions.csv'
ITEMS_CSV = 'data/items.csv'

# --- Process Reviews (Interactions) ---
print("Processing reviews...")
with open(REVIEWS_PATH, 'r', encoding='utf-8') as fin, open(INTERACTIONS_CSV, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['user_id', 'item_id', 'timestamp'])

    for line in fin:
        try:
            review = json.loads(line)
            user_id = review.get('reviewerID')
            item_id = review.get('asin')
            timestamp = review.get('unixReviewTime')
            if user_id and item_id and timestamp:
                writer.writerow([user_id, item_id, timestamp])
        except json.JSONDecodeError:
            continue  # skip corrupted lines

print(f"✅ Saved interactions to {INTERACTIONS_CSV}")

# --- Process Metadata (Items) ---
print("Processing item metadata...")
with open(META_PATH, 'r', encoding='utf-8') as fin, open(ITEMS_CSV, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['item_id', 'category', 'brand', 'title', 'description'])

    for line in fin:
        try:
            meta = json.loads(line)
            item_id = meta.get('asin')
            categories = meta.get('category', [])
            # Flatten if nested
            if isinstance(categories, list):
                if categories and isinstance(categories[0], list):
                    categories = categories[0]
                category = categories[0] if categories else ''
            else:
                category = categories
            brand = meta.get('brand', '')
            title = meta.get('title', '')
            description = meta.get('description', '')

            if item_id:
                writer.writerow([item_id, category, brand, title, description])
        except json.JSONDecodeError:
            continue  # skip corrupted lines

print(f"✅ Saved item metadata to {ITEMS_CSV}")

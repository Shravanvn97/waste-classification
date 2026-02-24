import os
import json
import shutil

dataset_dir = "dataset/TACO/data"
annotations_file = os.path.join(dataset_dir, "annotations.json")

output_dir = "dataset/taco_classification"
os.makedirs(output_dir, exist_ok=True)

# Load annotations
with open(annotations_file, 'r') as f:
    data = json.load(f)

# Build image lookup from batches
image_lookup = {}
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_lookup[file] = os.path.join(root, file)

# Map categories
categories = {cat['id']: cat['name'] for cat in data['categories']}

# Map image ids to filenames
image_id_to_file = {img['id']: os.path.basename(img['file_name']) for img in data['images']}

copied = 0

for ann in data['annotations']:
    cat_name = categories[ann['category_id']]
    img_file = image_id_to_file[ann['image_id']]

    if img_file in image_lookup:
        src = image_lookup[img_file]

        dst_folder = os.path.join(output_dir, cat_name)
        os.makedirs(dst_folder, exist_ok=True)

        dst = os.path.join(dst_folder, img_file)

        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1

print(f"Conversion done — copied {copied} images.")

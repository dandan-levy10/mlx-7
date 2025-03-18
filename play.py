import json
import os

# -----------------------

# with open("recipe-1m/layer2.json", "r") as f:
#     image_data = json.load(f)

# required_images = []
# for entry in image_data:
#     if "images" in entry and entry["images"]:
#         img = entry["images"][0]  # Take only the first image
#         img_path = f"val/{img['id'][:1]}/{img['id'][1:2]}/{img['id'][2:3]}/{img['id'][3:4]}/{img['id']}.jpg"
#         required_images.append(img_path)

# print("🔍 First 10 expected image paths:")
# for path in required_images[:10]:
#     print(path)



# ------------------------

# missing_ids = ["c97f5159d4", "2e030d7ea0", "7e6a3442ce", "23e8751aff", "788e2a0bfd"]

# # Load image mapping data
# with open("recipe-1m/layer2.json", "r") as f:
#     image_data = json.load(f)

# image_recipe_ids = {entry["id"] for entry in image_data}

# for img_id in missing_ids:
#     print(f"✅ {img_id} exists in layer2.json: {img_id in image_recipe_ids}")

# ------------------------
# # Check expected paths vs actual paths
# # Load layer2.json
# with open("recipe-1m/layer2.json", "r") as f:
#     image_data = json.load(f)

# # Check how the paths are stored in layer2.json
# for entry in image_data:
#     for img in entry["images"]:
#         if img["id"] in ["c97f5159d4", "2e030d7ea0", "7e6a3442ce"]:
#             expected_path = f"val/{img['id'][:1]}/{img['id'][1:2]}/{img['id'][2:3]}/{img['id'][3:4]}/{img['id']}.jpg"
#             print(f"Expected path: {expected_path}")

# ------------------------

# UPDATED SCRIPT TO VERIFY EXTRACTED VS. EXPECTED IMAGES 

# # Load the list of selected recipes from layer1_train_subset.json
# with open("processed-data/layer1_train_subset.json", "r") as f:
#     train_recipes = json.load(f)

# # Get the recipe IDs that we selected
# selected_recipe_ids = {recipe["id"] for recipe in train_recipes}

# # Load layer2.json and filter for only the first image from our selected validation recipes
# expected_image_ids = set()
# with open("recipe-1m/layer2.json", "r") as f:
#     image_data = json.load(f)
#     for entry in image_data:
#         if entry["id"] in selected_recipe_ids and entry["images"]:  # ✅ Only check images for our selected recipes
#             expected_image_ids.add(entry["images"][0]["id"])  # ✅ Only store the first image

# print(f"📌 Expected images in validation set for our dataset (first image per recipe): {len(expected_image_ids)}")

# # Get the actual extracted image IDs
# extracted_image_ids = set()
# train_image_folder = "processed-data/train_images"
# for root, dirs, files in os.walk(train_image_folder):
#     for filename in files:
#         if filename.endswith(".jpg"):
#             extracted_image_ids.add(filename.replace(".jpg", ""))  # Remove .jpg for comparison

# print(f"✅ Extracted train images: {len(extracted_image_ids)}")

# # Compare extracted images vs. expected images
# missing_images = expected_image_ids - extracted_image_ids
# extra_images = extracted_image_ids - expected_image_ids

# print(f"❌ Missing images: {len(missing_images)}")
# print(f"⚠️ Unexpected extra images extracted: {len(extra_images)}")

# # Save missing images for debugging
# with open("missing_images.log", "w") as f:
#     for img_id in missing_images:
#         f.write(f"{img_id}\n")

# print("✅ Debugging data saved to missing_images.log.")

# ---------------------------

import json

# Load expected image IDs
with open("processed-data/layer1_train_subset.json", "r") as f:
    train_recipes = json.load(f)

# Pick the first recipe and get its expected image
sample_recipe = train_recipes[0]
sample_recipe_id = sample_recipe["id"]

# Get expected image from layer2.json
with open("recipe-1m/layer2.json", "r") as f:
    image_data = json.load(f)

sample_image_id = None
for entry in image_data:
    if entry["id"] == sample_recipe_id and entry["images"]:
        sample_image_id = entry["images"][0]["id"]
        break

print(f"🎯 Sample Recipe ID: {sample_recipe_id}")
print(f"🔍 Expected Image ID: {sample_image_id}")

expected_path = f"processed-data/train_images/{sample_image_id[:1]}/{sample_image_id[1:2]}/{sample_image_id[2:3]}/{sample_image_id[3:4]}/{sample_image_id}"
print(f"📌 Expected image storage path: {expected_path}")

# -------------------------------

# Check the filenames of stored images- do they reflect image id format or recipe id format? 


image_filenames = []

# Recursively search for all image files in train_images
for root, dirs, files in os.walk("processed-data/train_images"):
    for filename in files:
        if filename.endswith(".jpg"):  # Ensure we only get images
            image_filenames.append(filename)

# Print first 10 images found
print("📂 First 10 stored image filenames:", image_filenames[:10])

# COMPARE EXTRACTED FILENAMES TO EXPECTED IMAGE IDS

with open("recipe-1m/layer2.json", "r") as f:
    image_data = json.load(f)

# Get first recipe in layer2.json
first_entry = image_data[0]
recipe_id = first_entry["id"]
first_image_id = first_entry["images"][0]["id"]  # First image for that recipe

print(f"✅ Recipe ID: {recipe_id}")
print(f"✅ Expected Image ID: {first_image_id}")
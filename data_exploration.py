import json
import pandas as pd
import random
import tarfile
import os 

def inspect_json(file_path, num_samples=5):
    """Load and inspect a few samples from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)  # Load JSON

    print(f"✅ Loaded {len(data)} entries from {file_path}")
    print(f"🔹 Data Type: {type(data)}")  # List or Dict

    # Print sample entries
    for i, entry in enumerate(data[:num_samples]):
        # print(f"\n🔍 Sample {i+1}:")
        # for key, value in entry.items():
        #     print(f"  {key}: {str(value)[:400]}")  # Limit output to avoid excessive text
        print(entry["title"])



def preview_json(file_path, num_samples=5):
    """Load and preview a JSON file as a DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)  # Load JSON
    
    df = pd.DataFrame(data[:num_samples])  # Convert to DataFrame
    print(f"✅ Loaded {len(data)} entries from {file_path}")
    
    return df

# Preview layer1.json
# df_layer1 = preview_json("recipe-1m/layer1.json", num_samples=5)

# Preview layer2.json
# df_layer2 = preview_json("recipe-1m/layer2.json", num_samples=5)

# Uncomment below to view as df (columns cut off in display)
# print(df_layer1)
# print(df_layer2)

def create_test_train_dataset(train_samples, test_samples, seed=42):
    """Creates train and test datasets efficiently by processing JSON in chunks."""
    
    # Check if the files already exist
    if os.path.exists("processed-data/layer1_train_subset.json") and os.path.exists("processed-data/layer1_test_subset.json"):
        print("✅ Dataset already exists. Skipping generation.")
        return

    # Set random seed for reproducibility
    random.seed(seed)

    # Load image mapping data as a set for quick lookup
    image_recipe_ids = set()
    with open("recipe-1m/layer2.json", "r") as f:
        image_data = json.load(f)
        for entry in image_data:
            image_recipe_ids.add(entry["id"])  # Add recipe IDs that have images

    # Stream layer1.json and filter recipes with images
    recipes_with_images = []
    with open("recipe-1m/layer1.json", "r") as f:
        recipes = json.load(f)
        for recipe in recipes:
            if recipe["id"] in image_recipe_ids and recipe["partition"] == "val":
                recipes_with_images.append(recipe)

    print(f"✅ Found {len(recipes_with_images)} recipes with images")

    # Shuffle once and split efficiently
    random.shuffle(recipes_with_images)
    train_recipes = recipes_with_images[:train_samples]
    test_recipes = recipes_with_images[train_samples:train_samples + test_samples]

    # Save filtered datasets
    with open("processed-data/layer1_train_subset.json", "w") as f:
        json.dump(train_recipes, f, indent=4)

    with open("processed-data/layer1_test_subset.json", "w") as f:
        json.dump(test_recipes, f, indent=4)

    print(f"✅ Created {train_samples} training and {test_samples} test recipes.")
    print(f"🔍 Final train dataset size: {len(train_recipes)}")
    print(f"🔍 Final test dataset size: {len(test_recipes)}")


# def extract_required_images(tar_path, image_output_dir, train_json, test_json, layer2_json):
#     """Extracts only the required images for train/test datasets from the .tar archive with corrected file paths."""

#     def load_recipe_ids(json_file):
#         with open(json_file, "r") as f:
#             recipes = json.load(f)
#         return {recipe["id"] for recipe in recipes}

#     train_ids = load_recipe_ids(train_json)
#     test_ids = load_recipe_ids(test_json)

#     train_output_dir = os.path.join(image_output_dir, "train_images")
#     test_output_dir = os.path.join(image_output_dir, "test_images")
#     os.makedirs(train_output_dir, exist_ok=True)
#     os.makedirs(test_output_dir, exist_ok=True)

#     # Load image paths from layer2.json
#     required_images = {}
#     with open(layer2_json, "r") as f:
#         image_data = json.load(f)
#         for entry in image_data:
#             recipe_id = entry["id"]
#             if recipe_id in train_ids or recipe_id in test_ids:
#                 for img in entry["images"]:
#                     # **Updated path format to match the .tar file structure**
#                     img_path = f"val/{img['id'][:1]}/{img['id'][1:2]}/{img['id'][2:3]}/{img['id'][3:4]}/{img['id']}"
#                     required_images[img_path] = "train" if recipe_id in train_ids else "test"

#     print(f"✅ Found {len(required_images)} images to extract.")

#     # Extract only missing images
#     with tarfile.open(tar_path, "r") as tar:
#         for member in tar.getmembers():
#             if member.name in required_images:
#                 target_dir = train_output_dir if required_images[member.name] == "train" else test_output_dir
#                 output_path = os.path.join(target_dir, os.path.basename(member.name))

#                 # **Print extracted image paths for debugging**
#                 print(f"📂 Extracting: {member.name} to {target_dir}")

#                 if not os.path.exists(output_path):
#                     tar.extract(member, target_dir)

#     print("✅ Image extraction complete.")


def extract_required_images(tar_path, image_output_dir, train_json, test_json, layer2_json):
    """Extracts only the required images from the .tar archive and logs extraction failures."""

    def load_recipe_ids(json_file):
        with open(json_file, "r") as f:
            recipes = json.load(f)
        return {recipe["id"] for recipe in recipes}

    train_ids = load_recipe_ids(train_json)
    test_ids = load_recipe_ids(test_json)

    train_output_dir = os.path.join(image_output_dir, "train_images")
    test_output_dir = os.path.join(image_output_dir, "test_images")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Load image paths from layer2.json
    required_images = {}
    with open(layer2_json, "r") as f:
        image_data = json.load(f)
        for entry in image_data:
            recipe_id = entry["id"]
            if recipe_id in train_ids or recipe_id in test_ids:
                if entry["images"]: # Extract only the first image
                    img = entry["images"][0]
                    img_path = f"val/{img['id'][:1]}/{img['id'][1:2]}/{img['id'][2:3]}/{img['id'][3:4]}/{img['id']}"
                    required_images[img_path] = "train" if recipe_id in train_ids else "test"

    print(f"✅ Found {len(required_images)} images to extract.")

    missing_files = []
    extracted_count = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name in required_images:
                target_dir = train_output_dir if required_images[member.name] == "train" else test_output_dir
                output_path = os.path.join(target_dir, os.path.basename(member.name))

                # Debugging: Print each image being extracted
                print(f"📂 Extracting: {member.name} to {target_dir}")

                try:
                    if not os.path.exists(output_path):
                        try:
                            tar.extract(member, target_dir)
                            extracted_count += 1
                            print(f"✅ Extracted: {member.name}")
                        except Exception as e:
                            print(f"❌ Error extracting {member.name}: {e}")
                    else:
                        print(f"⚠️ Skipped (already exists): {output_path}")
                except Exception as e:
                    print(f"❌ Error extracting {member.name}: {e}")
                    missing_files.append(member.name)

    print(f"✅ Successfully extracted {extracted_count} images.")
    print(f"❌ {len(missing_files)} images failed to extract. Check missing_images.log.")

    # Save missing image paths for debugging
    with open("missing_images.log", "w") as f:
        for missing in missing_files:
            f.write(f"{missing}\n")

    print("✅ Image extraction process complete.")



if __name__ == "__main__":

    # Create the json files 
    create_test_train_dataset(20000, 5000)

    # Extract sampled images to new file
    extract_required_images(
    tar_path="recipe-1m/recipe-1m-images-validation.tar",
    image_output_dir="processed-data",
    train_json="processed-data/layer1_train_subset.json",
    test_json="processed-data/layer1_test_subset.json",
    layer2_json="recipe-1m/layer2.json"
    )
    # # Inspect layer1.json
    # inspect_json("recipe-1m/layer1.json", num_samples= 100)

    # # Inspect layer2.json
    # inspect_json("recipe-1m/layer2.json")
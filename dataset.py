import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import re

# class RecipeDataset(Dataset):
#     def __init__(self, json_file, image_root, transform=None):
#         """
#         Args:
#             json_file (str): Path to the JSON file containing recipes.
#             image_root (str): Root directory for images.
#             transform (callable, optional): Image transformations.
#         """
#         with open(json_file, "r") as f:
#             self.data = json.load(f)

#         self.image_root = image_root
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Resize((224, 224)),  # Resize to CLIP input size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         recipe = self.data[idx]

#         # Process text
#         title = recipe["title"]
#         ingredients = ", ".join([ing["text"] for ing in recipe["ingredients"]])
#         text_input = f"{title}. Ingredients: {ingredients}."

#         # Construct hierarchical image path (keeping full filename)
#         image_id = recipe["id"]
#         image_path = f"{self.image_root}/{image_id[:1]}/{image_id[1:2]}/{image_id[2:3]}/{image_id[3:4]}/{image_id}.jpg"

#         if not os.path.exists(image_path):
#             print(f"❌ Missing image: {image_path}")  # Log missing images
#             return None  # Skip this entry

#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)

#         return text_input, image


class RecipeDataset(Dataset):
    def __init__(self, json_file, image_root, layer2_json, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file containing recipes.
            image_root (str): Root directory for images.
            layer2_json (str): Path to the JSON file containing image mappings.
            transform (callable, optional): Image transformations.
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])

        # **Preload layer2.json into a dictionary for fast lookup**
        with open(layer2_json, "r") as f:
            image_data = json.load(f)
        
        self.recipe_to_image = {
            entry["id"]: entry["images"][0]["id"]  # Only store first image per recipe
            for entry in image_data if entry["images"]
        }

    def __len__(self):
        return len(self.data)
    
    def clean_ingredient_text(self, text):
        # Remove numbers, fractions, and measurement words
        text = re.sub(r"\d+[/\d]*\s*", "", text)  # Remove numbers & fractions
        text = re.sub(r"\b(teaspoons?|tablespoons?|cups?|ounces?|cloves?|slices?|strips?|Tbsp|Tsp|ml|g|lbs?|cut|in|thirds|half|halves|minced|substitute|sliced|into|inch|-inch|whole|sheets|thawed|weight|divided|chopped|diced|peeled|crushed|ground|fresh|frozen|package|container|can|jar|drained|packed|rinsed|shredded|crumbled|leaves|sprigs|stalks|heads|fillets|loaves)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\([^)]*\)", "", text)  # Remove anything in parentheses

        # Remove hyphens and commas
        text = re.sub(r"[-,]", " ", text)  # Replace hyphens and commas with spaces

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def __getitem__(self, idx):
        recipe = self.data[idx]

        # Process text
        title = recipe["title"]
        ingredients = ", ".join([ing["text"] for ing in recipe["ingredients"]])
        text_input = f"{title}. Ingredients: {ingredients}."
        text_input = self.clean_ingredient_text(text_input) # Clean text input

        # Get the correct image ID
        image_id = self.recipe_to_image.get(recipe["id"], None)

        if image_id is None:
            print(f"❌ No image found for recipe ID: {recipe['id']}")
            return None  # Skip this entry

        # Construct correct image path using image ID
        image_path = f"{self.image_root}/{image_id[:1]}/{image_id[1:2]}/{image_id[2:3]}/{image_id[3:4]}/{image_id}"

        if not os.path.exists(image_path):
            print(f"❌ Missing image: {image_path}")  # Log missing images
            return None  # Skip this entry

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return text_input, image


# Example usage
train_dataset = RecipeDataset(
    json_file="processed-data/layer1_train_subset.json",
    image_root="processed-data/train_images/val",
    layer2_json="recipe-1m/layer2.json"
)

test_dataset = RecipeDataset(
    json_file="processed-data/layer1_test_subset.json",
    image_root="processed-data/test_images/val",
    layer2_json="recipe-1m/layer2.json"
)

# Check dataset sample
# text_sample, image_sample = train_dataset[100] if train_dataset is not None else None, None
# print(f"Sample Text: {text_sample}")
# print(f"Image Shape: {image_sample.shape}")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # ✅ Remove None values
    return zip(*batch) if batch else ([], [])   # ✅ Prevents empty batch errors

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)


# Print 5 random dataset samples
for _ in range(4):
    idx = random.randint(0, len(train_dataset) - 1)
    sample = train_dataset[idx]
    
    if sample:
        text_sample, image_sample = sample
        print(f"📜 Text: {text_sample}")
        print(f"🖼️ Image Shape: {image_sample.shape}")
        print("-" * 50)
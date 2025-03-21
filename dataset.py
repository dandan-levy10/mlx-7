import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import re
from transformers import CLIPProcessor
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
    def __init__(self, json_file, image_root, layer2_json, transform=None, num_negatives=3):
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
        self.num_negatives = num_negatives
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP input size
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]) # CLIP processor already normalises images
        ])
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        text = re.sub(r"\b(teaspoons?|tablespoons?|cups?|ounces?|cubes?|cloves?|slices?|strips?|Tbsp|Tsp|ml|g|lbs?|cut|in|thirds|half|halves|minced|substitute|sliced|into|inch|-inch|whole|sheets|thawed|weight|divided|chopped|diced|peeled|crushed|fresh|frozen|package|container|can|jar|drained|packed|rinsed|shredded|crumbled|leaves|sprigs|stalks|heads|fillets|loaves|pounds?|cans?|to|taste|fluid|containers?|or|pieces?|large|medium|small|trimmed|plus|and|quartered|wise|length|thinly|crosswise|grams?|dash|each)\b", "", text, flags=re.IGNORECASE)
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

        # Tokenise text input
        query_tokens = self.clip_processor(text=text_input, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"]
        pos_tokens = query_tokens.clone()

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

        pos_image = Image.open(image_path).convert("RGB")
        pos_image = self.transform(pos_image)
        pos_image = self.clip_processor(images=pos_image, return_tensors="pt", do_rescale=False)["pixel_values"].squeeze(0)
        neg_images_list = []
        neg_tokens_list = []
        used_indices = set()
        while len(neg_images_list) < self.num_negatives:
            neg_idx = random.randint(0, len(self.data) - 1)
            if neg_idx == idx or neg_idx in used_indices:
                continue

            neg_recipe = self.data[neg_idx]
            neg_title = neg_recipe["title"]
            neg_ingredients = ", ".join([ing["text"] for ing in neg_recipe["ingredients"]])
            neg_text_input = f"{neg_title}. Ingredients: {neg_ingredients}."
            neg_text_input = self.clean_ingredient_text(neg_text_input) # Clean text input
            neg_text_tokens = self.clip_processor(text=neg_text_input, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"]

            # Get the correct image ID
            neg_image_id = self.recipe_to_image.get(neg_recipe["id"], None)

            if neg_image_id is None:
                print(f"❌ No image found for recipe ID: {neg_recipe['id']}")
                continue

            # Construct correct image path using image ID
            neg_image_path = f"{self.image_root}/{neg_image_id[:1]}/{neg_image_id[1:2]}/{neg_image_id[2:3]}/{neg_image_id[3:4]}/{neg_image_id}"

            if not os.path.exists(neg_image_path):
                print(f"❌ Missing image: {neg_image_path}")  # Log missing images
                continue

            neg_image = Image.open(neg_image_path).convert("RGB")
            neg_image = self.transform(neg_image)
            neg_image = self.clip_processor(images=neg_image, return_tensors="pt", do_rescale=False)["pixel_values"].squeeze(0)

            neg_images_list.append(neg_image)
            neg_tokens_list.append(neg_text_tokens)
            used_indices.add(neg_idx)

        neg_tokens = torch.stack(neg_tokens_list, dim=0)
        neg_images = torch.stack(neg_images_list, dim=0)

        return query_tokens, pos_tokens, pos_image, neg_tokens, neg_images # Returns a tuple of (text_input, image, negative_samples)



# Check dataset sample
# text_sample, image_sample = train_dataset[100] if train_dataset is not None else None, None
# print(f"Sample Text: {text_sample}")
# print(f"Image Shape: {image_sample.shape}")

if __name__ == "__main__":

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

    # Print 5 random dataset samples
    for _ in range(1):
        idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[idx]
        
        if sample:
            query_tokens, pos_tokens, pos_image, neg_tokens, neg_images = sample
            print(f"📜 Text: {query_tokens['input_ids']}")
            print(f"🖼️ Image Shape: {pos_image.shape}")
            # print("image: ", pos_image)
            print(f"🖼️ Negative Samples: {len(neg_images)}")
            print("neg_images shape: ", neg_images[0].shape)
            print("neg_tokens: ", neg_tokens[0]["input_ids"])
            print("-" * 50)

    # print(f"✅ Total recipes mapped to images in old version: {len(train_dataset.recipe_to_image)}")
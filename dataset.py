import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class RecipeDataset(Dataset):
    def __init__(self, json_file, image_root, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file containing recipes.
            image_root (str): Root directory for images.
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data[idx]

        # Process text
        title = recipe["title"]
        ingredients = ", ".join([ing["text"] for ing in recipe["ingredients"]])
        text_input = f"{title}. Ingredients: {ingredients}."

        # Construct hierarchical image path (keeping full filename)
        image_id = recipe["id"]
        image_path = f"{self.image_root}/{image_id[:1]}/{image_id[1:2]}/{image_id[2:3]}/{image_id[3:4]}/{image_id}.jpg"

        if not os.path.exists(image_path):
            print(f"❌ Missing image: {image_path}")  # Log missing images
            return None  # Skip this entry

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return text_input, image



# Example usage
train_dataset = RecipeDataset(
    json_file="processed-data/layer1_train_subset.json",
    image_root="processed-data/train_images"
)

test_dataset = RecipeDataset(
    json_file="processed-data/layer1_test_subset.json",
    image_root="processed-data/test_images"
)

# Check dataset sample
text_sample, image_sample = train_dataset[100] if train_dataset is not None else None, None
# print(f"Sample Text: {text_sample}")
# print(f"Image Shape: {image_sample.shape}")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # ✅ Remove None values
    return zip(*batch) if batch else ([], [])   # ✅ Prevents empty batch errors

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)

# Print the first 5 dataset entries
for i in range(5):
    recipe = train_dataset.data[i]
    image_id = recipe["id"]
    image_path = f"{train_dataset.image_root}/{image_id[:1]}/{image_id[1:2]}/{image_id[2:3]}/{image_id[3:4]}/{image_id}.jpg"
    print(f"Recipe ID: {image_id} → Expected Image Path: {image_path}")


import json

with open("recipe-1m/layer2.json", "r") as f:
    image_data = json.load(f)

recipe_ids_with_images = {entry["id"] for entry in image_data}

# Check a specific ID
test_id = train_dataset.data[0]["id"]
print(f"✅ Exists in layer2.json? {'Yes' if test_id in recipe_ids_with_images else 'No'}")

# from transformers import CLIPTokenizer

# # Load CLIP tokenizer
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# # Tokenize "(1/2 cups)"
# tokens = tokenizer("(1/2 cups)", add_special_tokens=False)

# # Count number of tokens
# print(f"Number of tokens: {len(tokens.input_ids)}")


import json

# Load our filtered dataset
with open("processed-data/layer1_train_subset.json", "r") as f:
    train_data = json.load(f)

# Count how many of these recipes have image mappings in layer2.json
with open("recipe-1m/layer2.json", "r") as f:
    image_data = json.load(f)

image_recipe_ids = {entry["id"] for entry in image_data}

expected_images = sum(1 for recipe in train_data if recipe["id"] in image_recipe_ids)

print(f"Expected number of images for train dataset: {expected_images}")

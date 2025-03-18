import json
import random

# Load layer1.json (recipe metadata)
with open("layer1.json" , "r") as f:
    recipes = json.load(f)

# Split by partition (train/test/val) - I only downloaded val
train_recipes = [r for r in recipes if r["partition"] == "val"] # Using validation images
test_recipes = random.sample(train_recipes)
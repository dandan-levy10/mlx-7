import json

# from transformers import CLIPTokenizer

# # Load CLIP tokenizer
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# # Tokenize "(1/2 cups)"
# tokens = tokenizer("(1/2 cups)", add_special_tokens=False)

# # Count number of tokens
# print(f"Number of tokens: {len(tokens.input_ids)}")


with open("recipe-1m/layer2.json", "r") as f:
    image_data = json.load(f)

required_images = []
for entry in image_data:
    if "images" in entry and entry["images"]:
        img = entry["images"][0]  # Take only the first image
        img_path = f"val/{img['id'][:1]}/{img['id'][1:2]}/{img['id'][2:3]}/{img['id'][3:4]}/{img['id']}.jpg"
        required_images.append(img_path)

print("🔍 First 10 expected image paths:")
for path in required_images[:10]:
    print(path)


from transformers import CLIPTokenizer
import re





def clean_ingredient_text(text):
    # Remove numbers, fractions, and measurement words
    text = re.sub(r"\d+[/\d]*\s*", "", text)  # Remove numbers & fractions
    text = re.sub(r"\b(teaspoons?|tablespoons?|cups?|ounces?|cloves?|slices?|strips?|Tbsp|Tsp|ml|g|lbs?|cut|in|thirds|half|halves|minced|substitute|sliced|into|inch|-inch|whole|sheets|thawed|weight|divided|chopped|diced|peeled|crushed|ground|fresh|frozen|package|container|can|jar|drained|packed|rinsed|shredded|crumbled|leaves|sprigs|stalks|heads|fillets|loaves)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)  # Remove anything in parentheses

    # Remove hyphens and commas
    text = re.sub(r"[-,]", " ", text)  # Replace hyphens and commas with spaces

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Example usage
raw_ingredients = """
12 ounces chicken breast tenders, cut in thirds, 1 teaspoon cornstarch, 1 tablespoon low-sodium soy sauce, 
1/4 cup fresh lemon juice (start with 2 Tbsp, mix, taste and add more if you wish), 1/4 cup low-sodium soy sauce,
1/4 cup fat-free chicken broth, 1 teaspoon fresh ginger, minced, 2 garlic cloves, minced, 
2 teaspoons Splenda sugar substitute, 1 teaspoon cornstarch, 2 teaspoons vegetable oil, 
1/4 cup red bell pepper, sliced into 2-inch strips, 1/4 cup green bell pepper, sliced into 2-inch strips.
"""

cleaned_text = clean_ingredient_text(raw_ingredients)
print("🔹 Cleaned Ingredients:", cleaned_text)

# Load CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

sample_text = """Diabetic Asian Lemon Chicken. Ingredients: 12 ounces chicken breast tenders,
 cut in thirds, 1 teaspoon cornstarch, 1 tablespoon low sodium soy sauce, 14 cup fresh lemon
 juice (start with 2 Tbsp, mix, taste and add more if you wish), 14 cup low sodium soy sauce,
 14 cup fat free chicken broth, 1 teaspoon fresh ginger, minced, 2 garlic cloves, minced, 
 2 teaspoons Splenda sugar substitute, 1 teaspoon cornstarch, 2 teaspoons vegetable oil, 
 14 cup red bell pepper, sliced into 2-inch strips, 14 cup green bell pepper, sliced into 2-inch strips."""

sample_text = """Chocolate Banana Eggrolls. Ingredients: 4 whole Sheets Phyllo Dough, Thawed, 2 ounces, weight Bittersweet Chocolate, 1 whole Banana, 2 teaspoons Brown Sugar, Divided, 1 teaspoon Cinnamon, Divided, 13 cups Heavy Cream."""

# Tokenize "(1/2 cups)"
tokens = tokenizer(cleaned_text, add_special_tokens=False)

# Count number of tokens
print(f"Number of tokens: {len(tokens.input_ids)}")

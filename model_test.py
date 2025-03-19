from model import ThreeTowerModel, NTXentLoss
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = ThreeTowerModel(clip_model, clip_processor)

# Sample batch
batch_size = 2
num_negatives = 3

# Image transformations
transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP input size
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
        ])

# Example text and images
query = ["query recipe 1", "query recipe 2"]
pos_text = ["positive recipe 1", "positive recipe 2"]
pos_image = torch.randn(batch_size, 3, 224, 224)  # Simulated image tensors
pos_image = transform(pos_image)
pos_image = torch.clamp(pos_image, 0, 1)
neg_texts = [["neg1 recipe 1", "neg2 recipe 1", "neg3 recipe 1"],
             ["neg1 recipe 2", "neg2 recipe 2", "neg3 recipe 2"]]  # Shape: [B, N]
neg_images = torch.randn(batch_size, num_negatives, 3, 224, 224)  # Shape: [B, N, 3, 224, 224]
neg_images = transform(neg_images)
neg_images = torch.clamp(neg_images, 0, 1)
model.eval()

with torch.no_grad():
    query_embeddings, pos_embeddings, neg_embeddings = model(query, pos_text, pos_image, neg_texts, neg_images)

# Print shapes
print("✅ Query Embedding Shape:", query_embeddings.shape)  # Expected: [B, 512]
print("✅ Positive Text Embedding Shape:", pos_embeddings.shape)  # Expected: [B, 512]
print("✅ Positive Image Embedding Shape:", neg_embeddings.shape)  # Expected: [B, N, 512]

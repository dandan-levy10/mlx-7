import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RecipeDataset
from model import ThreeTowerModel, NTXentLoss
from utils import set_device
from transformers import CLIPModel, CLIPProcessor


def train(num_epochs=10):
    # Set device
    device = set_device()
    
    # Load dataset
    dataset = torch.utils.data.Subset(RecipeDataset(
        json_file="processed-data/layer1_train_subset.json",
        image_root="processed-data/train_images/val",
        layer2_json="recipe-1m/layer2.json",
        num_negatives=3
    ), range(10))

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize model
    model = ThreeTowerModel(clip_model=clip_model, clip_processor=clip_processor, hidden_dim=100).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Initialize loss function
    criterion = NTXentLoss()

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Get inputs and labels
            query_tokens, pos_tokens, pos_image, neg_tokens, neg_images = batch

            # Move inputs to device
            query_tokens = query_tokens.to(device)
            pos_tokens = pos_tokens.to(device)
            pos_image = pos_image.to(device)
            neg_tokens = neg_tokens.to(device)
            neg_images = neg_images.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            query_embeddings, pos_embeddings, neg_embeddings = model(query_tokens, pos_tokens, pos_image, neg_tokens, neg_images)

            # Compute loss
            loss = criterion(query_embeddings, pos_embeddings, neg_embeddings)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Print loss
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

if __name__ == "__main__":
    train(3)
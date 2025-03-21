import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeTowerModel(nn.Module):
    def __init__(self, clip_model, clip_processor, hidden_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.hidden_dim = hidden_dim

        # Freeze CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False 

        # Query tower
        self.query_tower = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Text tower
        self.text_tower = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Image tower
        self.image_tower = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Fusion tower
        self.fusion_tower = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(self, query_tokens, pos_tokens, pos_image, neg_tokens, neg_images):
        batch_size = len(query_tokens)

        # Preprocess positive image- not needed as this is already done in the dataset
        # pos_image_tensor = self.clip_processor(images=pos_image, return_tensors="pt")["pixel_values"] # [Batch_size, 3, 224, 224]

        # Encode query + positive samples
        query_embedding = self.clip_model.get_text_features(input_ids=query_tokens) # [Batch_size, 512]
        pos_text_embedding = self.clip_model.get_text_features(input_ids=pos_tokens) # [Batch_size, 512]
        pos_image_embedding = self.clip_model.get_image_features(pixel_values=pos_image) # [Batch_size, 512]

        # print("Query Embedding Shape:", query_embedding.shape)         # Should be [B, 512]
        # print("Positive Embedding Shape:", pos_text_embedding.shape)     # Should be [B, 512]
        
        # Tokenise & encode negative text samples
        neg_tokens = neg_tokens.view(-1, 77) # [Batch_size*neg_samples, 77] --> CLIP encoder expects 2D input
        neg_text_embedding = self.clip_model.get_text_features(input_ids=neg_tokens) # [Batch_size*neg_samples, 512]
        # print("Negatives (before reshape) Shape:", neg_text_embedding.shape)  # Should be [B*N, 512]
        neg_text_embedding = neg_text_embedding.view(batch_size, -1, 512) # Reshape to[Batch_size, neg_samples, 512]
        # print("Negatives (after reshape) Shape:", neg_text_embedding.shape)  # Should be [B, N, 512]


        # Tokenise & encode negative image samples- not needed as this is already done in the dataset
        # neg_image_tensor = self.clip_processor(images=neg_images, return_tensors="pt")["pixel_values"] # [Batch_size,neg_samples, 3, 224, 224]
        neg_images = neg_images.view(-1, 3, 224, 224) # [Batch_size*neg_samples, 3, 224, 224] --> CLIP encoder expects 4D input
        neg_image_embedding = self.clip_model.get_image_features(pixel_values=neg_images) # [Batch_size, neg_samples, 512]
        neg_image_embedding = neg_image_embedding.view(batch_size, -1, 512) # Reshape to[Batch_size, neg_samples, 512]

        # Pass through MLP towers
        query_embeddings = self.query_tower(query_embedding)  # [Batch_size, hidden_dim]
        pos_text_embeddings = self.text_tower(pos_text_embedding)  # [Batch_size, hidden_dim]
        pos_image_embeddings = self.image_tower(pos_image_embedding)  # [Batch_size, hidden_dim]

        # Process negative samples
        neg_text_embeddings = self.text_tower(neg_text_embedding)  # [Batch_size, neg_samples, hidden_dim]
        neg_image_embeddings = self.image_tower(neg_image_embedding)  # [Batch_size, neg_samples, hidden_dim]

        # Fuse text & image embeddings
        pos_embeddings = self.fusion_tower(torch.cat([pos_text_embeddings, pos_image_embeddings], dim=1)) # [Batch_size, hidden_dim]
        neg_embeddings = self.fusion_tower(torch.cat([neg_text_embeddings, neg_image_embeddings], dim=2)) # [Batch_size, num_negatives, hidden_dim]

        query_embeddings = F.normalize(query_embeddings, dim=-1)
        pos_embeddings = F.normalize(pos_embeddings, dim=-1)
        neg_embeddings = F.normalize(neg_embeddings, dim=-1)

        # print("End of forward pass query_embeddings shape: ", query_embeddings.shape)
        # print("End of forward pass neg_embeddings shape: ", neg_embeddings.shape)
        # print("End of forward pass pos_embeddings shape: ", pos_embeddings.shape)

        return query_embeddings, pos_embeddings, neg_embeddings
        
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings, pos_embeddings, neg_embeddings):
        batch_size = query_embeddings.size(0)

        # Compute similarities
        pos_similarity = torch.exp(torch.nn.functional.cosine_similarity(query_embeddings, pos_embeddings) / self.temperature)    # [Batch_size]
        # # Unsqueeze query_embeddings from [B, 512] to [B, 1, 512]
        neg_similarity = torch.exp(torch.matmul(query_embeddings.unsqueeze(1), neg_embeddings.transpose(1, 2)) / self.temperature)  # [Batch_size, num_negatives]
        # This gives a shape of [B, 1, neg_samples]; then squeeze it:
        neg_similarity = neg_similarity.squeeze(1).sum(dim=1)  # (B,1,N) --> (B,N) --> [Batch_size]
        
        # print("Loss function pos_similarity shape: ", pos_similarity.shape)
        # print("Loss function neg_similarity shape: ", neg_similarity.shape)

        # Compute denominator: sum of all positive and negative similarities
        denominator = pos_similarity + neg_similarity

        # Compute loss
        loss = -torch.log(pos_similarity / denominator).mean()

        return loss
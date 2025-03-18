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

    def forward(self, query, pos_text, pos_image, neg_texts, neg_images):
        
        # Preprocess query + positive inputs
        query_tokens = self.clip_processor(text=query, return_tensors="pt", padding=True, truncation=True)
        pos_text_tokens = self.clip_processor(text=pos_text, return_tensors="pt", padding=True, truncation=True)
        pos_image_tensor = self.clip_processor(images=pos_image, return_tensors="pt")["pixel_values"]

        # Preprocess negative inputs
        neg_text_tokens = torch.stack([self.clip_processor(text=neg, return_tensors="pt", padding=True, truncation=True) for neg in neg_texts])
        neg_image_tensor = torch.stack([self.clip_processor(images=neg, return_tensors="pt")["pixel_values"] for neg in neg_images])

        # Encode query + positive samples
        query_embedding = self.clip_model.get_text_features(**query_tokens)
        pos_text_embedding = self.clip_model.get_text_features(**pos_text_tokens)
        pos_image_embedding = self.clip_model.get_image_features(pos_image_tensor)

        # Encode negative samples
        neg_text_embedding = torch.stack([self.clip_model.get_text_features(neg) for neg in neg_text_tokens])
        neg_image_embedding = torch.stack([self.clip_model.get_image_features(neg) for neg in neg_image_tensor])

        # Pass through MLP towers
        query_tower_output = self.query_tower(query_embedding)
        pos_text_tower_output = self.text_tower(pos_text_embedding)
        pos_image_tower_output = self.image_tower(pos_image_embedding)

        # Process negative samples
        neg_text_tower_output = torch.stack([self.text_tower(neg) for neg in neg_text_embedding])
        neg_image_tower_output = torch.stack([self.image_tower(neg) for neg in neg_image_embedding])

        # Fuse text & image embeddings
        pos_fusion_tower_output = self.fusion_tower(torch.cat([pos_text_tower_output, pos_image_tower_output], dim=1))
        neg_fusion_tower_output = torch.stack([self.fusion_tower(torch.cat([neg_text_tower_output, neg_image_tower_output], dim=1)) for neg_text_tower_output, neg_image_tower_output in zip(neg_text_tower_output, neg_image_tower_output)])

        return query_tower_output, pos_fusion_tower_output, neg_fusion_tower_output
        
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embedding, pos_fusion_tower_output, neg_fusion_tower_output):
        batch_size = query_embedding.size(0)

        # Compute similarities

        pos_similarity = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(1), pos_fusion_tower_output.unsqueeze(0), dim=2)
        neg_similarity = torch.exp(torch.matmul(query_embedding, neg_fustion))
import torch
import torch.nn as nn

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

    def forward(self, query, pos_text, pos_image, neg_text, neg_image):
        
        # Preprocess text inputs
        query_tokens = self.clip_processor(text=query, return_tensors="pt", padding=True, truncation=True)
        pos_text_tokens = self.clip_processor(text=pos_text, return_tensors="pt", padding=True, truncation=True)
        neg_text_tokens = self.clip_processor(text=neg_text, return_tensors="pt", padding=True, truncation=True)

        # Preprocess image inputs
        pos_image_tensor = self.clip_processor(images=pos_image, return_tensors="pt")["pixel_values"]
        neg_image_tensor = self.clip_processor(images=neg_image, return_tensors="pt")["pixel_values"]

        # Encode text & images
        query_embedding = self.clip_model.get_text_features(**query_tokens)
        pos_text_embedding = self.clip_model.get_text_features(**pos_text_tokens)
        neg_text_embedding = self.clip_model.get_text_features(**neg_text_tokens)
        pos_image_embedding = self.clip_model.get_image_features(pos_image_tensor)
        neg_image_embedding = self.clip_model.get_image_features(neg_image_tensor)

        # Pass through MLP towers
        query_tower_output = self.query_tower(query_embedding)
        pos_text_tower_output = self.text_tower(pos_text_embedding)
        neg_text_tower_output = self.text_tower(neg_text_embedding)
        pos_image_tower_output = self.image_tower(pos_image_embedding)
        neg_image_tower_output = self.image_tower(neg_image_embedding)

        # Fuse text & image embeddings
        pos_fusion_tower_output = self.fusion_tower(torch.cat([pos_text_tower_output, pos_image_tower_output], dim=1))
        neg_fusion_tower_output = self.fusion_tower(torch.cat([neg_text_tower_output, neg_image_tower_output], dim=1))

        return query_tower_output, pos_fusion_tower_output, neg_fusion_tower_output
        
        
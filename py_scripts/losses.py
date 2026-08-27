import torch
import torch.nn as nn
import torch.nn.functional as F


# stable with softmax
class FastSupCon_v2(nn.Module):
    def __init__(self, temperature: float = 0.07, num_classes: int = 2, ignore_index=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, N, C], labels: [B, N] or [B, H, W] -> .view(B, -1)
        B, N, C = features.size()
        labels = labels.view(B, -1)
        
        # 1. feature normalization
        features = F.normalize(features, p=2, dim=-1).float()
        
        # 2. Kernel trick
        # Temperature is removed so it doesn't corrupt kernel representations
        phi = F.elu(features) + 1.0  
        phi_normalized = F.normalize(phi, p=2, dim=-1)

        # 3. Ignore mask generation
        if self.ignore_index is not None:
            valid_mask = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_classes)
            clean_labels = labels.clone()
            clean_labels[~valid_mask] = 0
            one_hot_labels = F.one_hot(clean_labels.long(), num_classes=self.num_classes).float()
            one_hot_labels = one_hot_labels * valid_mask.unsqueeze(-1)
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
            one_hot_labels = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        
        # 4. Centroids calculation
        class_counts = one_hot_labels.sum(dim=1, keepdim=True) # [B, 1, K]
        # Class mask that shows which classes appears in the image
        class_present_mask = (class_counts > 0).float() # [B, 1, K]
        
        class_sums = torch.bmm(one_hot_labels.transpose(1, 2), phi_normalized) # [B, K, C]
        # adding epsilon to prevent division by zero
        class_means = class_sums / (class_counts.transpose(1, 2) + 1e-8) # [B, K, C]
        class_means_normalized = F.normalize(class_means, p=2, dim=-1)
        
        # 5. Similarity calculation (cosine similarity)
        # result in [-1, 1]
        similarity = torch.bmm(phi_normalized, class_means_normalized.transpose(1, 2)) # [B, N, K]
        
        # Apply temperature
        logits = similarity / self.temperature # [B, N, K]
        
        # 6. softmax
        log_prob = F.log_softmax(logits, dim=-1) # [B, N, K]
        
        # probability of a pixel to be the same class that it is already
        per_pixel_loss = -(log_prob * one_hot_labels).sum(dim=-1) # [B, N]
        
        # removing invalid pixels from calculation
        per_pixel_loss = per_pixel_loss * valid_mask.float()
        
        # Making sure that we count only valid pixels
        num_valid_pixels = valid_mask.sum()
        if num_valid_pixels < 10: 
            return features.sum() * 0.0
            
        return per_pixel_loss.sum() / num_valid_pixels



# stable with margin
class FastSupCon(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 0.3, num_classes: int = 2, ignore_index=None):
        super().__init__()
        self.temperature = temperature
        self.margin = margin 
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, N, C = z.size()
        labels = labels.view(B, -1)
        
        # 1. eature normalization
        z_norm = F.normalize(z, p=2, dim=-1).float()
        
        # 2. Kernel trick
        phi = F.elu(z_norm) + 1.0  
        phi_normalized = F.normalize(phi, p=2, dim=-1)

        # 3. ignore mask
        if self.ignore_index is not None:
            valid_mask = (labels != self.ignore_index) & (labels >= 0) & (labels < self.num_classes)
            clean_labels = labels.clone()
            clean_labels[~valid_mask] = 0
            one_hot_labels = F.one_hot(clean_labels.long(), num_classes=self.num_classes).float()
            one_hot_labels = one_hot_labels * valid_mask.unsqueeze(-1)
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)
            one_hot_labels = F.one_hot(labels.long(), num_classes=self.num_classes).float()
        
        # 4. centroids calculations
        class_counts = one_hot_labels.sum(dim=1, keepdim=True) # [B, 1, K]
        
        class_sums = torch.bmm(one_hot_labels.transpose(1, 2), phi_normalized) # [B, K, C]
        class_means = class_sums / (class_counts.transpose(1, 2) + 1e-8) # [B, K, C]
        class_means_normalized = F.normalize(class_means, p=2, dim=-1)
        
        # 5. similarity calculation [-1, 1]
        similarity = torch.bmm(phi_normalized, class_means_normalized.transpose(1, 2)) # [B, N, K]
        
        # applying temperature
        scaled_similarity = similarity / self.temperature

        # part with margin
        S_pos = (scaled_similarity * one_hot_labels).sum(dim=-1) # [B, N]
        
        # getting similarity with the strongest wrong class
        inverse_labels = 1.0 - one_hot_labels
        # decreasing value of pixel true class to remove it from possible max output
        masked_norm = scaled_similarity * inverse_labels + (one_hot_labels * -10000.0)
        S_neg_max, _ = torch.max(masked_norm, dim=-1) # [B, N]
        
        # hinge loss
        loss_flat = F.relu(self.margin - (S_pos - S_neg_max))
        
        # removing invalid pixels
        loss_flat = loss_flat * valid_mask.float()
        
        num_valid_pixels = valid_mask.sum()
        if num_valid_pixels < 10: 
            return z.sum() * 0.0
            
        return loss_flat.sum() / num_valid_pixels

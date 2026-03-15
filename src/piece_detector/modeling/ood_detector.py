"""
Mahalanobis OOD Detector for DINO model.

This detector uses Mahalanobis distance to identify out-of-distribution objects.
- Low distance = in-distribution (known chess piece)
- High distance = out-of-distribution (hand, unknown object)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class MahalanobisOODDetector(nn.Module):
    """
    OOD detector using Mahalanobis distance on decoder features.
    
    Calibration phase:
        1. Run model on training images
        2. Collect decoder features for each detected class
        3. Compute mean vector per class and shared covariance matrix
        
    Inference phase:
        1. For each detection, compute Mahalanobis distance to all class means
        2. Use minimum distance as OOD score
        3. High score = likely OOD
    """
    
    def __init__(self, num_classes: int = 13, feature_dim: int = 256, device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        
        # Calibration statistics (will be computed during calibration)
        self.register_buffer("class_means", torch.zeros(num_classes, feature_dim))
        self.register_buffer("precision_matrix", torch.eye(feature_dim))
        self.register_buffer("is_calibrated", torch.tensor(False))
        
        # For collecting features during calibration
        self.class_features: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_classes)}
        
    def collect_features(self, features: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, 
                         min_score: float = 0.7):
        """
        Collect decoder features for calibration.
        
        Args:
            features: Decoder features [N, feature_dim]
            labels: Predicted class labels [N]
            scores: Detection scores [N]
            min_score: Minimum score to include (only high-confidence detections)
        """
        mask = scores >= min_score
        features = features[mask]
        labels = labels[mask]
        
        for feat, label in zip(features, labels):
            label_int = label.item()
            if 0 <= label_int < self.num_classes:
                self.class_features[label_int].append(feat.detach().cpu())
    
    def compute_statistics(self):
        """
        Compute class means and shared precision matrix from collected features.
        """
        all_features = []
        class_means = torch.zeros(self.num_classes, self.feature_dim)
        class_counts = torch.zeros(self.num_classes)
        
        print("\nComputing OOD statistics...")
        for class_id in range(self.num_classes):
            features = self.class_features[class_id]
            if len(features) > 0:
                stacked = torch.stack(features)
                class_means[class_id] = stacked.mean(dim=0)
                class_counts[class_id] = len(features)
                all_features.append(stacked)
                print(f"  Class {class_id}: {len(features)} samples")
            else:
                print(f"  Class {class_id}: 0 samples (will use zero mean)")
        
        # Compute shared covariance matrix
        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
            print(f"  Total samples: {len(all_features)}")
            
            # Center features by subtracting per-class means
            centered_features = []
            idx = 0
            for class_id in range(self.num_classes):
                n = int(class_counts[class_id].item())
                if n > 0:
                    class_feats = all_features[idx:idx+n]
                    centered = class_feats - class_means[class_id]
                    centered_features.append(centered)
                    idx += n
            
            if len(centered_features) > 0:
                centered_features = torch.cat(centered_features, dim=0)
                
                # Compute covariance matrix
                cov_matrix = torch.mm(centered_features.T, centered_features) / (len(centered_features) - 1)
                
                # Add small regularization for numerical stability
                cov_matrix = cov_matrix + 1e-4 * torch.eye(self.feature_dim)
                
                # Compute precision matrix (inverse of covariance)
                precision_matrix = torch.linalg.inv(cov_matrix)
            else:
                precision_matrix = torch.eye(self.feature_dim)
        else:
            precision_matrix = torch.eye(self.feature_dim)
        
        # Store results
        self.class_means.copy_(class_means.to(self.device))
        self.precision_matrix.copy_(precision_matrix.to(self.device))
        self.is_calibrated.fill_(True)
        
        print("  Calibration complete!")
        
    def compute_mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from features to each class mean.
        Returns minimum distance (distance to nearest class).
        
        Args:
            features: [N, feature_dim] decoder features
            
        Returns:
            distances: [N] minimum Mahalanobis distance for each feature
        """
        if not self.is_calibrated:
            raise RuntimeError("OOD detector not calibrated! Run calibration first.")
        
        # Move everything to the same device as features
        device = features.device
        class_means = self.class_means.to(device)
        precision_matrix = self.precision_matrix.to(device)
        
        # Compute distance to each class mean
        distances = []
        for class_id in range(self.num_classes):
            diff = features - class_means[class_id].unsqueeze(0)  # [N, D]
            # Mahalanobis: sqrt(diff @ precision @ diff.T)
            left = torch.mm(diff, precision_matrix)  # [N, D]
            dist_sq = (left * diff).sum(dim=1)  # [N]
            dist = torch.sqrt(torch.clamp(dist_sq, min=0))  # [N]
            distances.append(dist)
        
        # Stack and take minimum (distance to nearest class)
        distances = torch.stack(distances, dim=1)  # [N, num_classes]
        min_distances, _ = distances.min(dim=1)  # [N]
        
        return min_distances
    
    def is_ood(self, features: torch.Tensor, threshold: float = 50.0) -> torch.Tensor:
        """
        Determine if detections are OOD based on Mahalanobis distance.
        
        Args:
            features: [N, feature_dim] decoder features
            threshold: Distance threshold (higher = more permissive)
            
        Returns:
            is_ood: [N] boolean tensor, True if OOD
        """
        distances = self.compute_mahalanobis_distance(features)
        return distances > threshold
    
    def save(self, path: str):
        """Save calibration statistics."""
        torch.save({
            "class_means": self.class_means.cpu(),
            "precision_matrix": self.precision_matrix.cpu(),
            "is_calibrated": self.is_calibrated.cpu(),
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
        }, path)
        print(f"Saved OOD detector to {path}")
    
    def load(self, path: str):
        """Load calibration statistics."""
        data = torch.load(path, map_location=self.device)
        self.class_means.copy_(data["class_means"].to(self.device))
        self.precision_matrix.copy_(data["precision_matrix"].to(self.device))
        self.is_calibrated.fill_(True)
        print(f"Loaded OOD detector from {path}")


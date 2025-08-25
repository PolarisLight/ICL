import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMining_Proto(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.9, temperature=0.07, k=3, device='cuda'):

        super(HardNegativeMining_Proto, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature
        self.k = k
        self.device = device

        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, feature_dim, device=device), requires_grad=False
        )

        self.register_buffer("confusion_matrix", torch.zeros(num_classes, num_classes, device=device))

    def update_prototypes(self, features, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            feature_mean = features[mask].mean(dim=0).detach()
            self.prototypes.data[label] = (
                self.momentum * self.prototypes.data[label] +
                (1 - self.momentum) * feature_mean
            )

    def update_confusion_matrix(self, predicted, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        predicted = predicted.long()
        labels = labels.long()
        indices = torch.stack([labels, predicted], dim=0)
        updates = torch.ones_like(labels, dtype=torch.float)
        self.confusion_matrix.index_put_(tuple(indices), updates, accumulate=True)

    def apply_epoch_momentum(self, momentum=0.9):

        self.confusion_matrix *= 1 - momentum

    def get_hard_negative_classes(self, labels):

        hard_negatives = []
        for label in labels:
            class_confusion = self.confusion_matrix[label] 
            _, hard_classes = torch.topk(class_confusion, k=self.k, largest=True)
            hard_negatives.append(hard_classes)
        return torch.stack(hard_negatives)

    def compute_contrastive_loss(self, features, labels):

        if labels.dim() == 2 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        pos_sim = F.cosine_similarity(features, prototypes[labels], dim=1)

        hard_negative_classes = self.get_hard_negative_classes(labels)  # (batch_size, k)
        hard_negative_prototypes = prototypes[hard_negative_classes]  # (batch_size, k, feature_dim)

        neg_sim = torch.matmul(
            features.unsqueeze(1),  # (batch_size, 1, feature_dim)
            hard_negative_prototypes.transpose(1, 2)  # (batch_size, feature_dim, k)
        ).squeeze(1) / self.temperature  # (batch_size, k)

        weighted_neg_sim = torch.exp(neg_sim).mean(dim=1)

        loss = -torch.log(torch.exp(pos_sim / self.temperature) / (torch.exp(pos_sim / self.temperature) + weighted_neg_sim))
        return loss.mean()

    def reset_confusion_matrix(self):

        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)


import torch
import torch.nn.functional as F


def euclidean(source: torch.Tensor, target: torch.Tensor) -> float:
    pdist = torch.nn.PairwiseDistance(p=2)
    return pdist(source, target)


def cosine_similarity(source: torch.Tensor, target: torch.Tensor) -> float:
    return F.cosine_similarity(source, target, dim=0)


def dot_product(source: torch.Tensor, target: torch.Tensor) -> float:
    return torch.dot(source.double(), target.double())
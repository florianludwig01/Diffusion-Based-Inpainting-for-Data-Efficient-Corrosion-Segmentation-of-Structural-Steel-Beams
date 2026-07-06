import torch

def rust_iou_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > 0.5

    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float()

    if union == 0:
        return torch.tensor(0.0)

    return intersection / union
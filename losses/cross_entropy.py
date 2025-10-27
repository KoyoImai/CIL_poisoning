

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union



class SeenCrossEntropyLoss(nn.Module):
    
    """
    Cross-entropy over *seen classes* for single-head CIL.

    Expected model output y_pred is a LIST of per-task logits tensors,
    where each item has shape [B, K_t] and concatenating the first (t+1)
    items yields the logits over all classes seen up to current task t.

    Usage:
        criterion = SeenCrossEntropyLoss()
        loss = criterion(y_pred, targets, task_id)

    - y_pred: List[Tensor]  (output of model(images))
    - targets: Tensor[ B ]  (global class ids consistent with concatenation order)
    - task_id: int          (0-based current task index)
    """
    
    def __init__(self):
        super().__init__()

    def forward(self,
                y_pred: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
                targets: torch.Tensor,
                task_id: int) -> torch.Tensor:
        
        if isinstance(y_pred, (list, tuple)):
            assert len(y_pred) > 0, "y_pred list is empty"
            # clamp task_id to available heads
            t = min(task_id, len(y_pred) - 1)
            # concat logits for tasks [0..t] => seen classes
            logits_seen = torch.cat(list(y_pred[:t+1]), dim=1)
            return F.cross_entropy(logits_seen, targets)
        
        else:
            raise TypeError(
                "SeenCrossEntropyLoss expects the model output to be a list/tuple "
                "of per-task logits. Got type: {}".format(type(y_pred))
            )






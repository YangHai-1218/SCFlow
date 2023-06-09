import torch
import torch.nn as nn
from torch import Tensor

from .builder import LOSSES, build_loss


@LOSSES.register_module()
class RAFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, max_flow=400, eps=1e-10):
        super().__init__()
        self.loss_weight = loss_weight
        self.max_flow = max_flow
        self.eps = eps
    
    def forward(self, pred_flow:torch.Tensor, gt_flow:torch.Tensor, valid=None):
        mag = torch.sum(gt_flow**2, dim=1).sqrt()
        if valid is None:
            valid = (mag < self.max_flow).to(gt_flow)
        else:
            valid = ((valid >= 0.5) & (mag < self.max_flow)).to(gt_flow)
        loss = (pred_flow - gt_flow).abs()
        loss = (valid[:, None] * loss).sum() / (valid.sum() + self.eps)
        return self.loss_weight * loss



@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-10):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps 
    
    def forward(self, pred_mask:torch.Tensor, gt_mask:torch.Tensor, valid=None):
        loss = torch.mean(torch.abs(pred_mask - gt_mask))
        return loss * self.loss_weight



@LOSSES.register_module()
class SequenceLoss(nn.Module):
    """Sequence Loss for RAFT.
    Args:
        gamma (float): The base of exponentially increasing weights. Default to
            0.8.
        max_flow (float): The maximum value of optical flow, if some pixel's
            flow of target is larger than it, this pixel is not valid.
                Default to 400.
    """

    def __init__(self, loss_func_cfg:dict, gamma: float = 0.8, ) -> None:
        super().__init__()
        self.loss_func = build_loss(loss_func_cfg)
        self.gamma = gamma
    
    def to(self, device):
        self.loss_func.to(device)

    def forward(self,
                *preds,
                **kwargs) -> Tensor:
        """Forward function for MultiLevelEPE.
        Args:
            preds_dict Sequence[Tensor]: The list of predicted optical flow.
            target (Tensor): Ground truth of optical flow with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for optical flow.
                Defaults to None.
        Returns:
            Tensor: value of pixel-wise end point error loss.
        """
        n_preds = len(preds[0])
        loss = 0.
        seq_loss_list = []
        for i in range(n_preds):
            i_weight = self.gamma **(n_preds - i -  1)
            i_loss = self.loss_func(*[pred_ele[i] for pred_ele in preds], **kwargs)
            loss += i_weight * i_loss
            seq_loss_list.append(i_loss)
        
        return loss, seq_loss_list
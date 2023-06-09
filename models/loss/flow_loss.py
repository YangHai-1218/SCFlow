from typing import Optional
import torch






def endpoint_error(pred: torch.Tensor,
                   target: torch.Tensor,
                   p: int = 2,
                   q: Optional[float] = None,
                   eps: Optional[float] = None) -> torch.Tensor:
    r"""Calculate end point errors between prediction and ground truth.

    If not define q, the loss function is
    .. math::
      loss = \Vert \mathbf{u}-\mathbf{u_gt} \Vert^p

    otherwise,
    .. math::
      loss = (\Vert \mathbf{u}-\mathbf{u_gt} \Vert^p+eps)^q

    For PWC-Net L2 norm loss: p=2, for the robust loss function p=1, q=0.4,
    eps=0.01.

    Args:
        pred (torch.Tensor): output flow map from flow_estimator
            shape(B, 2, H, W).
        target (torch.Tensor): ground truth flow map shape(B, 2, H, W).
        p (int): norm degree for loss. Options are 1 or 2. Defaults to 2.
        q (float, optional): used to give less penalty to outliers when
            fine-tuning model. Defaults to 0.4.
        eps (float, optional): a small constant to numerical stability when
            fine-tuning model. Defaults to 0.01.

    Returns:
        Tensor: end-point error map with the shape (B, H, W).
    """

    assert pred.shape == target.shape, \
        (f'pred shape {pred.shape} does not match target '
         f'shape {target.shape}.')

    epe_map = torch.norm(pred - target, p, dim=1)  # shape (B, H, W).

    if q is not None and eps is not None:
        epe_map = (epe_map + eps)**q

    return epe_map






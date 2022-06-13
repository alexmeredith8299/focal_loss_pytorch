"""
This function implements binary focal loss for tensors of arbitrary size/shape.
"""
import warnings
import torch

class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the binary focal loss between each element
    in the input and target tensors.

    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean

    Attributes
    -----------
        gamma: float (optional)
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean'):
        if reduction not in ("sum", "mean", "none"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.

        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask

        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        #Warn that if sizes don't match errors may occur
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size"\
                "({input_tensor.size()}). \n This will likely lead to incorrect results"\
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )
        #Broadcast to get sizes/shapes to match
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of binary focal loss
        pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
        loss_tensor = -(1-pt_tensor)**self.gamma*torch.log(pt_tensor)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            return torch.mean(loss_tensor)
        #If not none or mean, sum
        return torch.sum(loss_tensor)

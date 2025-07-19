import torch.nn.functional as F
import torch
import cv2


def mean_diff(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the difference in peak location b/w the two histograms.

    Parameters:
    -----------
    input   : ground truth image (1, ..., num_bins)
    pred    : predicted image for each particle (batch_size, ..., num_bins)

    Returns:
    --------
    mode_diff   : difference in peak locations (batch_size, )

    """
    num_bins = input.shape[-1]
    x = torch.arange(num_bins).reshape(1, 1, 1, num_bins).to(pred.device)

    # === Normalize temporal dimension to have integral 1 === #
    input /= torch.linalg.norm(input + 1e-8, dim=-1, keepdim=True)
    pred /= torch.linalg.norm(pred + 1e-8, dim=-1, keepdim=True)

    # === Compute expected value (mean) along time axis === #
    input_mean = torch.sum(input * x, dim=-1)
    pred_mean = torch.sum(pred * x, dim=-1)

    # === Compute mean difference === #
    mean_difference = torch.nanmean(torch.abs(input_mean - pred_mean), dim=tuple(range(1, input_mean.dim()))) 
    mean_score = (num_bins - mean_difference) / num_bins

    # # === Compute varaince along time === #
    # input_var = torch.sum(input * (x - input_mean.unsqueeze(-1))**2, dim=-1)
    # pred_var = torch.sum(pred * (x - pred_mean.unsqueeze(-1))**2, dim=-1)

    # # === Compute variance difference === #
    # var_difference = torch.abs(input_var - pred_var).mean(dim=tuple(range(1, input_var.dim())))

    return mean_score# + var_difference 


def gradient_correlation(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    corr   : average dot product of gradients along space and time axes (batch_size, )

    """

    # === Compute gradients === #
    input_grad_y = input[:, 1:, :, :] - input[:, :-1, :, :]
    input_grad_x = input[:, :, 1:, :] - input[:, :, :-1, :]
    input_grad_t = input[:, :, :, 1:] - input[:, :, :, :-1]

    pred_grad_y = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_grad_t = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    # # === Normalize gradients === #
    dims = tuple(range(1, pred.dim()))
    pred_grad_x /= torch.linalg.vector_norm(pred_grad_x, dim=dims, keepdim=True) + 1e-7
    pred_grad_y /= torch.linalg.vector_norm(pred_grad_y, dim=dims, keepdim=True) + 1e-7
    pred_grad_t /= torch.linalg.vector_norm(pred_grad_t, dim=dims, keepdim=True) + 1e-7

    # === Compute dot product of gradients === #
    dot_x = torch.sum(input_grad_x * pred_grad_x, dim=dims)
    dot_y = torch.sum(input_grad_y * pred_grad_y, dim=dims)
    dot_t = torch.sum(input_grad_t * pred_grad_t, dim=dims)
    
    # === Compute average === #
    corr = (1/3) * (dot_x + dot_y + dot_t)

    return corr

def mse_score(input: torch.Tensor, pred: torch.Tensor, k: float = None) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    loss   : mean squared error loss (batch_size, )

    """
    if k is None:
        # k = 5 / input.shape[-1]
        k = 1 / input.shape[-1]
    loss = torch.mean((input - pred)**2, dim=tuple(range(1, input.dim()))) # (batch_size, )
    print(torch.min(loss), torch.max(loss))
    score = torch.exp(-k * loss)
    # score = -loss

    return score


def dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    # input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)
    # input /= torch.sum(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.sum(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score

def normalized_dot_product_score(gt_meas: torch.Tensor, rend_meas: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    gt_img   : measured space-time image (1, num_pixels, num_bins)
    rend_img : predicted image for each particle (batch_size, num_pixels, num_bins)

    Returns:
    --------
    score   : normalized dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    gt_img_norm = torch.linalg.vector_norm(gt_meas.clone(), dim=tuple(range(1, gt_meas.dim())), keepdim=True)
    gt_img = gt_meas.clone() / (gt_img_norm + 1e-7)

    rend_img_norm = torch.linalg.vector_norm(rend_meas, dim=tuple(range(1, rend_meas.dim())), keepdim=True)
    rend_img = rend_meas.clone() / (rend_img_norm + 1e-7)
    # input /= torch.sum(input, dim=tuple(range(1, input.dim())), keepdim=True)
    # pred /= torch.sum(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(gt_img * rend_img, dim=tuple(range(1, gt_meas.dim())))

    return score

def filtered_dot_product_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : dot product of input and pred (batch_size, )
    """

    # === Normalize images === #
    input /= torch.linalg.vector_norm(input, dim=tuple(range(1, input.dim())), keepdim=True)
    pred /= torch.linalg.vector_norm(pred, dim=tuple(range(1, pred.dim())), keepdim=True)

    # === Compute dot product === #
    score = torch.sum(input * pred, dim=tuple(range(1, input.dim())))

    return score


def tof_diff(input : torch.Tensor, pred : torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : difference in peak location along time axis (batch_size, )

    """
    # === Compute tof === #
    input_tof = torch.argmax(input, dim=-1) # (1, n_y, n_x)
    pred_tof = torch.argmax(pred, dim=-1) # (batch_size, n_y, n_x)

    # import matplotlib.pyplot as plt
    # for i in range(input.shape[1]):
    #     for j in range(input_tof.shape[2]):
    #         plt.plot(input[0, i, j].cpu().numpy(), 'r')
    #         plt.plot(pred[0, i, j].cpu().numpy(), 'b')
    #         plt.legend(['input', 'pred'])
    #         plt.show()
    # print("input", input_tof)
    # print("pred", pred_tof)

    # print(input_tof.shape, pred_tof.shape)

    # === Compute difference === #
    diff = torch.abs(input_tof - pred_tof).float().mean(dim=tuple(range(1, input_tof.dim()))) 
    score = input.shape[-1] - diff

    # print(torch.min(score), torch.max(score))
    return score

def weighted_score(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
    -----------
    input   : ground truth image (1, n_y, n_x, num_bins)
    pred    : predicted image for each particle (batch_size, n_y, n_x, num_bins)

    Returns:
    --------
    score   : weighted score (batch_size, )

    """
    tof_score = mean_diff(input, pred)
    corr_score = dot_product_score(input, pred)
    score = 0.5 * (tof_score + corr_score)

    return score
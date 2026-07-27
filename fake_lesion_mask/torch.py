from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as nd


@dataclass(frozen=True)
class FLMParams:

    options: tuple[str, ...] = ('as_new_lesion', 'stable', 'erosion', 'dilation')
    p: tuple[float, ...] | None = None

    dil_iters: tuple[int, ...] = (1, 2, 3)
    dil_p_iters: tuple[float, ...] = (8/12, 3/12, 1/12)

    ero_iters: tuple[int, ...] = (1, 2, 3)
    ero_p_iters: tuple[float, ...] = (8/12, 3/12, 1/12)



def fast_gpu_fake_lesion_mask(lesion_mask: torch.Tensor,
                              structure: np.ndarray | None,
                              params: FLMParams = FLMParams()
                              ):

    if not torch.any(lesion_mask).item():
        return torch.zeros_like(lesion_mask)

    device = lesion_mask.device

    mask_np = lesion_mask.cpu().numpy()
    if mask_np.ndim >= 3 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)
    labeled_mask_np, num_lesions = nd.label(mask_np, structure)

    # Immediately back to GPU
    labeled_tensor = torch.from_numpy(labeled_mask_np).to(device)
    del mask_np, labeled_mask_np

    stable_ids = []
    erode_iters = []  # tuples of (id, iterations)
    dilate_iters = [] # tuples of (id, iterations)

    ops = np.random.choice(params.options, num_lesions, replace=True, p=params.p)
    for i, op in enumerate(ops, start=1):

        if op == 'as_new_lesion':
            continue # doing nothing naturally drops it from the final mask
        elif op == 'stable':
            stable_ids.append(i)
        elif op == 'erosion':
            iters = np.random.choice(params.ero_iters, p=params.ero_p_iters)
            erode_iters.append((i, iters))
        elif op == 'dilation':
            iters = np.random.choice(params.dil_iters, p=params.dil_p_iters)
            dilate_iters.append((i, iters))

    pool_op = getattr(F, f'max_pool{lesion_mask.ndim}d')
    pool_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    final_mask = torch.zeros_like(lesion_mask, dtype=torch.bool)

    # stable lesions
    if stable_ids:
        stable_mask = torch.isin(labeled_tensor, torch.tensor(stable_ids, device=device))
        final_mask |= stable_mask

    # dil
    for current_iter in params.dil_iters:
        ids_this_iter = [i for i, it in dilate_iters if it == current_iter]
        if ids_this_iter:
            # Isolate all lesions needing this exact iteration count
            base_mask = torch.isin(labeled_tensor, torch.tensor(ids_this_iter, device=device)).float()

            # Add batch and channel dims for 3D pooling: [1, 1, D, H, W]
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)

            # Fast cuDNN native dilation
            for _ in range(current_iter):
                base_mask = pool_op(base_mask, **pool_kwargs)

            final_mask |= base_mask.squeeze(0).squeeze(0).bool()

    # ero
    for current_iter in params.ero_iters:
        ids_this_iter = [i for i, it in erode_iters if it == current_iter]
        if ids_this_iter:
            base_mask = torch.isin(labeled_tensor, torch.tensor(ids_this_iter, device=device)).float()
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)

            # Fast cuDNN native erosion (Max pooling the inverted mask)
            for _ in range(current_iter):
                base_mask = 1.0 - pool_op(1.0 - base_mask, **pool_kwargs)

            final_mask |= base_mask.squeeze(0).squeeze(0).bool()

    return final_mask

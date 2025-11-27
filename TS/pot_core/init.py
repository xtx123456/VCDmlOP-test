import math
import torch
import torch.nn as nn

def gmm_init_(tensor: torch.Tensor, fan_in: int) -> None:
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    sigma = math.sqrt(2.0 / (5.0 * fan_in))
    mu = 2.0 * sigma
    numel = tensor.numel()
    with torch.no_grad():
        comp = torch.randint(0, 2, (numel,), device=tensor.device, dtype=torch.int8)
        means = torch.where(
            comp == 1,
            torch.full((numel,), mu, device=tensor.device),
            torch.full((numel,), -mu, device=tensor.device),
        )
        samples = torch.randn(numel, device=tensor.device) * sigma + means
        tensor.copy_(samples.view_as(tensor))

def apply_pot_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        gmm_init_(module.weight, fan_in)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        fan_in = module.in_features
        gmm_init_(module.weight, fan_in)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
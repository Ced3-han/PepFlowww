import math
import torch


def tor_expmap(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (x + u) % (2 * math.pi)

def tor_logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(y - x), torch.cos(y - x))

def tor_projx(x: torch.Tensor) -> torch.Tensor:
    return x % (2 * math.pi)

def tor_random_uniform(*size, dtype=None, device=None) -> torch.Tensor:
    z = torch.rand(*size, dtype=dtype, device=device)
    return z * 2 * math.pi

def tor_uniform_logprob(x):
    dim = x.shape[-1]
    return torch.full_like(x[..., 0], -dim * math.log(2 * math.pi))

def tor_geodesic_t(t, angles_1, angles_0):
    # target, base
    tangent_vec = t * tor_logmap(angles_0, angles_1)
    points_at_time_t = tor_expmap(angles_0, tangent_vec)
    return points_at_time_t

if __name__ =='__main__':
    a = tor_random_uniform((2,3,5))
    b = tor_random_uniform((2,3,5))
    t = torch.ones((2,1)) * 0.2
    c = tor_geodesic_t(t[...,None],a,b)
    print(c)
    print(c.shape)
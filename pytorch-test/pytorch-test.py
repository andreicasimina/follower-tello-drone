import torch

x = torch.rand(5, 3)

print(f'\nRandom 5x3 Matrix: \n', x)

cuda_available = torch.cuda.is_available()

print(f'\nIs cuda available: \n', cuda_available)
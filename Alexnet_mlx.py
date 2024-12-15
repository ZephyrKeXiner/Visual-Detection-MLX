import mlx.core as mx
import mlx.nn as nn

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

class AlexnetMLX(nn.Module):
  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()
    self.layers=[
      nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      Flatten(),
      nn.Linear(384 * 6 * 6, 4096), nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096), nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, output_dim)
    ]

  def __call__(self, x):
    for l in self.layers:
      x = l(x)
    return x
"""
HybridCNNExtractor: SB3 feature extractor for PatchEnv's 263D observation.
  obs[:7]   -> scalar MLP  -> 64D
  obs[7:]   -> tiny CNN    -> 64D    (16x16 local occupancy grid)
  concat    -> Linear(128, 256)      -> 256D output to pi/vf heads
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

N_SCALARS = 7
GRID_SIZE = 16


class HybridCNNExtractor(BaseFeaturesExtractor):
    """
    Split-input feature extractor for hybrid scalar + spatial observation.

    Inputs:
      - Scalar features (7D): s_norm, ey, speed, psi_error, a, b, curvature
      - Local occupancy grid (256D = 16x16): rotated to patch heading

    Processing:
      - Scalar branch: simple MLP (7 -> 64)
      - Grid branch: tiny CNN (1x16x16 -> 64)
      - Fusion: concatenate + project to 256D

    Output: 256D features fed to PPO's pi/vf heads
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Scalar branch ---
        # Simple MLP for Frenet/shape features
        self.scalar_net = nn.Sequential(
            nn.Linear(N_SCALARS, 64),
            nn.ReLU(),
        )

        # --- CNN branch ---
        # Tiny CNN for local spatial context
        # Input: (B, 1, 16, 16) - grayscale local occupancy grid
        # Layer 1: detect wall edges (stride=2 halves spatial size)
        #   (B, 1, 16, 16) -> (B, 8, 8, 8)
        # Layer 2: detect corners/corridors
        #   (B, 8, 8, 8) -> (B, 16, 4, 4)
        # Flatten: (B, 16*4*4=256)
        # Linear: compress to 64D
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
        )

        # --- Fusion head ---
        # Combine scalar (64D) + spatial (64D) = 128D, project to output dimension
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (batch, 263) float tensor from VecNormalize
                 First 7 values: scalars (s_norm, ey, speed, psi_error, a, b, curvature)
                 Last 256 values: flattened 16x16 grid

        Returns:
            features: (batch, features_dim) tensor fed to actor/critic heads
        """
        # Split observation back into components
        scalars = obs[:, :N_SCALARS]  # (batch, 7)
        grid_flat = obs[:, N_SCALARS:]  # (batch, 256)

        # Reshape flat grid to image format expected by Conv2d: (batch, channels, height, width)
        grid_img = grid_flat.view(-1, 1, GRID_SIZE, GRID_SIZE)  # (batch, 1, 16, 16)

        # Run each branch
        scalar_features = self.scalar_net(scalars)  # (batch, 64)
        grid_features = self.cnn(grid_img)  # (batch, 64)

        # Concatenate and fuse
        combined = torch.cat([scalar_features, grid_features], dim=1)  # (batch, 128)
        return self.fusion(combined)  # (batch, 256)

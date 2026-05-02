"""
HybridCNNExtractor: SB3 feature extractor for PatchEnv's 273D observation.
  obs[:17]  -> scalar MLP  -> 64D   (7 state scalars + 10 curvature lookahead values)
  obs[17:]  -> tiny CNN    -> 64D   (16x16 local occupancy grid)
  concat    -> Linear(128, 256)     -> 256D output to pi/vf heads
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

N_SCALARS = 17   # 7 state + 10 curvature lookahead (s+2m … s+20m)
GRID_SIZE = 16


class HybridCNNExtractor(BaseFeaturesExtractor):
    """
    Split-input feature extractor for hybrid scalar + spatial observation.

    Inputs:
      - Scalar features (17D):
          [0]  s_norm         — normalised arc-length (position on track)
          [1]  ey_signed      — cross-track error (m)
          [2]  speed          — patch longitudinal speed (m/s)
          [3]  psi_error      — heading error vs track tangent (rad)
          [4]  a              — patch major axis (m)
          [5]  b              — patch minor axis (m)
          [6]  curvature      — track curvature at current position (1/m)
          [7-16] curvature at s+2m, s+4m, …, s+20m  (10 lookahead values)
      - Local occupancy grid (256D = 16x16): map-frame, no rotation

    Processing:
      - Scalar branch: MLP (17 -> 64)  — learns speed-from-curvature policy
      - Grid branch: tiny CNN (1x16x16 -> 64)  — immediate obstacle avoidance
      - Fusion: concatenate + project to 256D

    Output: 256D features fed to PPO's pi/vf heads
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Scalar branch ---
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
        Args:
            obs: (batch, 273) float tensor from VecNormalize
                 First 17 values: scalars (state + curvature lookahead)
                 Last 256 values: flattened 16x16 occupancy grid
        Returns:
            features: (batch, features_dim) tensor fed to actor/critic heads
        """
        scalars = obs[:, :N_SCALARS]   # (batch, 17)
        grid_flat = obs[:, N_SCALARS:] # (batch, 256)

        # Reshape flat grid to image format expected by Conv2d: (batch, channels, height, width)
        grid_img = grid_flat.view(-1, 1, GRID_SIZE, GRID_SIZE)  # (batch, 1, 16, 16)

        # Run each branch
        scalar_features = self.scalar_net(scalars)  # (batch, 64)
        grid_features = self.cnn(grid_img)  # (batch, 64)

        # Concatenate and fuse
        combined = torch.cat([scalar_features, grid_features], dim=1)  # (batch, 128)
        return self.fusion(combined)  # (batch, 256)

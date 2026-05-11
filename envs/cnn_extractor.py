"""
HybridCNNExtractor: SB3 feature extractor for PatchEnv's 86D observation.
  obs[:22]  -> scalar MLP  -> 64D   (7 state + 10 curvature lookahead + 5 width lookahead)
  obs[22:]  -> tiny CNN    -> 64D   (8x8 local occupancy grid, 1m/pixel)
  concat    -> Linear(128, 256)     -> 256D output to pi/vf heads
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

N_SCALARS = 22   # 7 state + 10 curvature lookahead + 5 width lookahead
GRID_SIZE = 8


class HybridCNNExtractor(BaseFeaturesExtractor):
    """
    Split-input feature extractor for hybrid scalar + spatial observation.

    Inputs:
      - Scalar features (22D):
          [0]  s_norm              — normalised arc-length (position on track)
          [1]  ey_signed           — cross-track error (m)
          [2]  speed               — patch longitudinal speed (m/s)
          [3]  psi_error           — heading error vs track tangent (rad)
          [4]  a                   — patch major axis (m)
          [5]  b                   — patch minor axis (m)
          [6]  curvature           — track curvature at current position (1/m)
          [7-16] curvature at s+2m, s+4m, …, s+20m  (10 lookahead values)
          [17-21] track half-width at s+2m, s+5m, s+10m, s+15m, s+20m (5 values)
      - Local occupancy grid (64D = 8x8): map-frame, 1m/pixel, no rotation

    Processing:
      - Scalar branch: MLP (22 -> 64)  — learns curvature/width-aware speed policy
      - Grid branch:   CNN (1x8x8 -> 64) — immediate obstacle avoidance
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
        # Input: (B, 1, 8, 8) — 8x8 occupancy grid, 1m/pixel
        # Conv2d(1, 16, 3, stride=2, pad=1): (B,1,8,8) -> (B,16,4,4)
        # Flatten: (B, 256)
        # Linear: compress to 64D
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
        )

        # --- Fusion head ---
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, 86) float tensor from VecNormalize
                 First 22 values: scalars (state + curvature + width lookahead)
                 Last 64 values:  flattened 8x8 occupancy grid
        Returns:
            features: (batch, features_dim) tensor fed to actor/critic heads
        """
        scalars   = obs[:, :N_SCALARS]   # (batch, 22)
        grid_flat = obs[:, N_SCALARS:]   # (batch, 64)

        grid_img = grid_flat.view(-1, 1, GRID_SIZE, GRID_SIZE)  # (batch, 1, 8, 8)

        scalar_features = self.scalar_net(scalars)   # (batch, 64)
        grid_features   = self.cnn(grid_img)          # (batch, 64)

        combined = torch.cat([scalar_features, grid_features], dim=1)  # (batch, 128)
        return self.fusion(combined)  # (batch, 256)

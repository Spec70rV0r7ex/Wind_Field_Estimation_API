import torch
import numpy as np
from config.settings import settings

def create_sar_tiles(vv_db: np.ndarray, grid_shape: tuple) -> torch.Tensor:
    """
    Converts a flat array of SAR data back into an image grid,
    then slices it into tiles (e.g., 20x20) for ResNet ingestion.
    """
    vv_grid = vv_db.reshape(grid_shape)
    vv_norm = (vv_grid - np.mean(vv_grid)) / np.std(vv_grid)
    num_tiles = grid_shape[0] * grid_shape[1]
    tiles_tensor = torch.tensor(vv_norm, dtype=torch.float32).view(num_tiles, 1, 1, 1)
    tiles_tensor = tiles_tensor.expand(num_tiles, 1, settings.TILE_SIZE, settings.TILE_SIZE)

    return tiles_tensor

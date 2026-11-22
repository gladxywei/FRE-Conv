"""Inference helpers used by the public evaluation entrypoint."""

import torch


def spatial_tiled_inference(network, image, tile_size=128, halo=32):
    """Run a shape-preserving HSI model in spatial tiles.

    Every tile keeps the complete spectral axis. Only height and width are
    divided, and symmetric spatial halos provide context around each core.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected [N, C, B, H, W], found {tuple(image.shape)}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")

    height, width = image.shape[-2:]
    if tile_size >= height and tile_size >= width:
        return network(image)

    output = torch.empty_like(image)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        input_top = max(0, top - halo)
        input_bottom = min(height, bottom + halo)
        core_top = top - input_top
        core_bottom = core_top + bottom - top
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            input_left = max(0, left - halo)
            input_right = min(width, right + halo)
            core_left = left - input_left
            core_right = core_left + right - left
            tile = image[..., input_top:input_bottom, input_left:input_right]
            prediction = network(tile)
            if prediction.shape != tile.shape:
                raise ValueError(
                    f"Tiled model must preserve shape: input={tuple(tile.shape)}, "
                    f"output={tuple(prediction.shape)}"
                )
            output[..., top:bottom, left:right] = prediction[
                ..., core_top:core_bottom, core_left:core_right
            ]
    return output

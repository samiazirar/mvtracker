# Copyright (c) ETH VLG.
# Licensed under the terms in the LICENSE file at the root of this repo.

from pathlib import Path
import os
import torch

_WEIGHTS = {
    "mvtracker_main": "hf://ethz-vlg/mvtracker::mvtracker_200000_june2025.pth",
    "mvtracker_cleandepth": "hf://ethz-vlg/mvtracker::mvtracker_200000_june2025_cleandepth.pth",
}


def _load_ckpt(spec: str):
    if spec.startswith("http"):
        return torch.hub.load_state_dict_from_url(spec, map_location="cpu")
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        repo_id, filename = spec[len("hf://"):].split("::", 1)
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=os.getenv("HF_TOKEN"))
        return torch.load(path, map_location="cpu")
    path = Path(spec).expanduser().resolve()
    return torch.load(str(path), map_location="cpu")


def _extract_model_state(sd):
    """
    Accept:
      - plain state dict
      - {'state_dict': ...}
      - {'model': ..., 'optimizer': ..., 'scheduler': ..., 'total_steps': ...}
    Returns a clean model state_dict.
    """
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    # Strip optional "model." prefix
    sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def _build_model(**overrides):
    from mvtracker.models.core.mvtracker.mvtracker import MVTracker
    cfg = dict(
        sliding_window_len=12,
        stride=4,
        normalize_scene_in_fwd_pass=False,
        fmaps_dim=128,
        add_space_attn=True,
        num_heads=6,
        hidden_size=256,
        space_depth=6,
        time_depth=6,
        num_virtual_tracks=64,
        use_flash_attention=True,
        corr_n_groups=1,
        corr_n_levels=4,
        corr_neighbors=16,
        corr_add_neighbor_offset=True,
        corr_add_neighbor_xyz=False,
        corr_filter_invalid_depth=False,
    )
    cfg.update(overrides)
    return MVTracker(**cfg)


def _load_into(model, checkpoint_key: str):
    raw = _load_ckpt(_WEIGHTS[checkpoint_key])
    sd = _extract_model_state(raw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")
    return model


def mvtracker_model(*,
                    pretrained: bool = False,
                    device: str = "cuda",
                    checkpoint: str = "mvtracker_main",
                    **model_kwargs):
    """
    Return a bare MVTracker nn.Module.

    - pretrained=False: random init with model_kwargs.
    - pretrained=True : load from _WEIGHTS[checkpoint], then .eval().
    """
    model = _build_model(**model_kwargs).to(device)
    if pretrained:
        model = _load_into(model, checkpoint)
        model.eval()
    return model


def mvtracker_predictor(*,
                        pretrained: bool = True,
                        device: str = "cuda",
                        checkpoint: str = "mvtracker_main",
                        model_kwargs: dict | None = None,
                        predictor_kwargs: dict | None = None):
    """
    Return EvaluationPredictor wrapped around MVTracker.

    Pass model configuration via `model_kwargs={...}` (matches MVTracker.__init__).
    Pass predictor configuration via `predictor_kwargs={...}`:
      - interp_shape, visibility_threshold, grid_size, n_grids_per_view,
        local_grid_size, local_extent, sift_size, num_uniformly_sampled_pts, n_iters
    """
    from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor

    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    predictor_kwargs = {} if predictor_kwargs is None else dict(predictor_kwargs)

    predictor_defaults = dict(
        interp_shape=(384, 512),
        visibility_threshold=0.5,
        grid_size=4,
        n_grids_per_view=1,
        local_grid_size=18,
        local_extent=50,
        sift_size=0,
        num_uniformly_sampled_pts=0,
        n_iters=6,
    )
    pk = {**predictor_defaults, **predictor_kwargs}

    model = mvtracker_model(pretrained=pretrained, device=device, checkpoint=checkpoint, **model_kwargs)
    return EvaluationPredictor(multiview_model=model, **pk)


def mvtracker(pretrained: bool = True, device: str = "cuda"):
    """Default public endpoint: predictor with main checkpoint."""
    return mvtracker_predictor(pretrained=pretrained, device=device, checkpoint="mvtracker_main")


def mvtracker_cleandepth(pretrained: bool = True, device: str = "cuda"):
    """Predictor with 'clean depth only' checkpoint."""
    return mvtracker_predictor(pretrained=pretrained, device=device, checkpoint="mvtracker_cleandepth")
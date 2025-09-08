# experimental/vision_sac_catalog.py

from dataclasses import asdict, is_dataclass
from typing import List
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import CNNEncoderConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT


class _ProjectToFixed(nn.Module):
    def __init__(self, base: nn.Module, in_dim: int, out_dim: int = 1024):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(in_dim, out_dim)
        self.output_dims = out_dim
    def forward(self, x):
        out = self.base(x)
        z = out[ENCODER_OUT]
        out[ENCODER_OUT] = self.proj(z)
        return out


class VisionSACCatalog(SACCatalog):
    def __init__(self, *args, **kwargs):
        # Accept BOTH old (config=RLModuleConfig) and new styles, and avoid double-passing model_config.
        if "config" in kwargs:
            cfg = kwargs.pop("config")
            obs_space = getattr(cfg, "observation_space", None)
            act_space = getattr(cfg, "action_space", None)
            model_cfg = getattr(cfg, "model_config", None)
        else:
            # observation_space
            if "observation_space" in kwargs:
                obs_space = kwargs.pop("observation_space")
            else:
                obs_space, *args = args
            # action_space
            if "action_space" in kwargs:
                act_space = kwargs.pop("action_space")
            else:
                act_space, *args = args
            # model_config or model_config_dict
            if "model_config_dict" in kwargs:
                model_cfg = kwargs.pop("model_config_dict")
            elif "model_config" in kwargs:
                model_cfg = kwargs.pop("model_config")
            elif len(args):
                model_cfg, *args = args
            else:
                model_cfg = None

        if is_dataclass(model_cfg):
            model_cfg = asdict(model_cfg)

        # Pass ONLY model_config_dict to base to avoid duplication.
        super().__init__(obs_space, act_space, model_config_dict=model_cfg, **kwargs)
        self.obs_space = self.observation_space  # some paths look for this alias

    # -------- helpers --------
    def _is_image_obs(self) -> bool:
        s = self.observation_space
        return isinstance(s, gym.spaces.Box) and len(s.shape) == 3 and s.dtype in (np.uint8, np.float32, np.float64)

    def _mc(self) -> dict:
        mc = getattr(self, "model_config_dict", None) or getattr(self, "model_config", None)
        return mc or {}

    def _to_cnn_filter_specifiers(self) -> List[List[int]]:
        mc = self._mc()
        convs = mc.get("conv_filters")
        if convs is None:
            cfs = mc.get("cnn_filter_specifiers")
            if cfs is not None:
                return cfs
            return [[32, [5, 5], 2], [64, [4, 4], 2], [128, [3, 3], 2], [128, [3, 3], 2]]
        spec = []
        for out_ch, k, s in convs:
            if isinstance(k, int):
                k = [k, k]
            spec.append([out_ch, k, s])
        return spec

    def _make_cnn_cfg(self, input_dims):
        mc = self._mc()
        return CNNEncoderConfig(
            input_dims=list(input_dims),                 # H, W, C*
            cnn_filter_specifiers=self._to_cnn_filter_specifiers(),
            cnn_use_bias=True,
            cnn_activation=mc.get("conv_activation", "relu"),
            cnn_use_layernorm=False,
            cnn_kernel_initializer=None,
            cnn_kernel_initializer_config=None,
            cnn_bias_initializer=None,
            cnn_bias_initializer_config=None,
            flatten_at_end=True,
        )

    def _build_and_project(self, framework: str, input_dims):
        enc = self._make_cnn_cfg(input_dims).build(framework=framework)
        h, w, c = input_dims
        dummy = torch.zeros((1, h, w, c), dtype=torch.float32)
        with torch.no_grad():
            out = enc({Columns.OBS: dummy})
            d_in = int(out[ENCODER_OUT].shape[-1])
        return _ProjectToFixed(enc, in_dim=d_in, out_dim=1024)

    # -------- encoders --------
    def build_encoder(self, framework: str):
        if self._is_image_obs():
            h, w, c = self.observation_space.shape
            return self._build_and_project(framework, (h, w, c))
        return super().build_encoder(framework)

    def build_qf_encoder(self, framework: str):
        if self._is_image_obs():
            h, w, c = self.observation_space.shape
            act_dim = int(np.prod(self.action_space.shape or ()))
            return self._build_and_project(framework, (h, w, c + act_dim))
        return super().build_qf_encoder(framework)

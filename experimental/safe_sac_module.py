from typing import Any, Dict
import torch
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT

class SafeSACTorchRLModule(DefaultSACTorchRLModule):
    def _concat_obs_act_channels(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Only tile for images; vector obs go through plain concat on last dim.
        if obs.ndim == 4 and actions.ndim == 2:
            C = int(self.catalog.observation_space.shape[-1])
            if obs.shape[-1] == C:  # NHWC
                B, H, W, _ = obs.shape
                a_img = actions.view(B, 1, 1, -1).expand(-1, H, W, -1)
                return torch.cat((obs, a_img), dim=-1)
            elif obs.shape[1] == C:  # NCHW
                B, _, H, W = obs.shape
                a_img = actions.view(B, -1, 1, 1).expand(-1, -1, H, W)
                return torch.cat((obs, a_img), dim=1)
        # Fallback / vector case:
        return torch.cat((obs, actions), dim=-1)

    # Match current RLlib signature exactly.
    def _qf_forward_train_helper(self, batch: Dict[str, Any], encoder, head, squeeze: bool = True):
        sa = self._concat_obs_act_channels(batch[Columns.OBS], batch[Columns.ACTIONS])
        enc_out = encoder({Columns.OBS: sa})
        q = head(enc_out[ENCODER_OUT])
        if squeeze and q.ndim == 2 and q.shape[-1] == 1:
            q = q.squeeze(-1)
        return q

    def forward_target(self, batch: Dict[str, Any]):
        sa = self._concat_obs_act_channels(batch[Columns.OBS], batch[Columns.ACTIONS])
        enc = getattr(self, "target_qf_encoder", None) or getattr(self, "qf_encoder")
        head = getattr(self, "target_qf", None) or getattr(self, "qf")
        enc_out = enc({Columns.OBS: sa})
        q = head(enc_out[ENCODER_OUT])
        if q.ndim == 2 and q.shape[-1] == 1:
            q = q.squeeze(-1)
        return q
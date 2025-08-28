from typing import Any, Dict, TYPE_CHECKING

import tree  # pip install dm_tree

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import ModuleID
import gymnasium as gym
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.core.learner.torch.torch_learner import TorchLearner

torch, nn = try_import_torch()
import torch.nn.functional as F

def _get_action_feat_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(space.nvec.sum())
    if isinstance(space, gym.spaces.Box):
        return int(torch.tensor(space.shape).prod().item())
    raise ValueError(f"Unsupported action space: {space}")

def _action_features(actions: torch.Tensor, space: gym.Space) -> torch.Tensor:
    if isinstance(space, gym.spaces.Discrete):
        return F.one_hot(actions.long(), num_classes=space.n).float()
    if isinstance(space, gym.spaces.MultiDiscrete):
        parts = []
        a = actions.long()
        for i, n in enumerate(space.nvec.tolist()):
            parts.append(F.one_hot(a[..., i], num_classes=int(n)).float())
        return torch.cat(parts, dim=-1)
    if isinstance(space, gym.spaces.Box):
        a = actions.float().view(actions.size(0), -1)
        low  = torch.as_tensor(space.low,  device=a.device, dtype=a.dtype).view(1, -1)
        high = torch.as_tensor(space.high, device=a.device, dtype=a.dtype).view(1, -1)
        scale = torch.where((high - low) > 0, (high - low), torch.ones_like(high))
        return 2.0 * (a - low) / scale - 1.0  # normalize to [-1,1]
    raise ValueError(f"Unsupported action space: {space}")
    
class IntrinsicCuriosityModel(TorchRLModule, SelfSupervisedLossAPI):
    "An implementation of the Intrinsic Curiosity Module (ICM)."

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, hiddens, act_name: str):
        layers, last = [], in_dim
        for h in hiddens:
            layers.append(nn.Linear(last, h))
            if act_name not in (None, "linear"):
                layers.append(get_activation_fn(act_name, "torch")())
            last = h
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _get_action_feat_dim(space):
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        if isinstance(space, gym.spaces.MultiDiscrete):
            return int(space.nvec.sum())
        if isinstance(space, gym.spaces.Box):
            return int(torch.tensor(space.shape).prod().item())
        raise ValueError(f"Unsupported action space: {space}")

    @staticmethod
    def _action_features(actions, space):
        if isinstance(space, gym.spaces.Discrete):
            return F.one_hot(actions.long(), num_classes=space.n).float()
        if isinstance(space, gym.spaces.MultiDiscrete):
            parts, a = [], actions.long()
            for i, n in enumerate(space.nvec.tolist()):
                parts.append(F.one_hot(a[..., i], num_classes=int(n)).float())
            return torch.cat(parts, dim=-1)
        if isinstance(space, gym.spaces.Box):
            a = actions.float().view(actions.size(0), -1)
            low  = torch.as_tensor(space.low,  device=a.device, dtype=a.dtype).view(1, -1)
            high = torch.as_tensor(space.high, device=a.device, dtype=a.dtype).view(1, -1)
            scale = torch.where((high - low) > 0, (high - low), torch.ones_like(high))
            return 2.0 * (a - low) / scale - 1.0
        raise ValueError(f"Unsupported action space: {space}")

    @override(TorchRLModule)
    def setup(self):
        cfg = self.model_config
        self._feature_dim = int(cfg.get("feature_dim", 512))   # <- single source of truth
        self._a_feat_dim  = _get_action_feat_dim(self.action_space)

        # Forward: [φ, a] -> φ'
        self._forward_net = self._mlp(
            self._feature_dim + self._a_feat_dim,  # 515 if feature_dim=512 and Box(3,)
            self._feature_dim,
            cfg.get("forward_net_hiddens", (256, 256)),
            cfg.get("forward_net_activation", "relu"),
        )

        # Inverse: [φ, φ'] -> a
        self._inverse_net = self._mlp(
            2 * self._feature_dim,
            self._a_feat_dim,
            cfg.get("inverse_net_hiddens", (256, 256)),
            cfg.get("inverse_net_activation", "relu"),
        )

        self._phi_proj = None  # created lazily to map encoder latents -> feature_dim




    def _flatten2(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1) if x.dim() > 2 else x
    
    def _to_channels_only(self, x):
        if x.dim() == 4:  # [B, C, H, W]
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, C]
        elif x.dim() == 3:  # [B, C, L]
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, C]
        elif x.dim() > 3:
            x = x.mean(dim=tuple(range(2, x.dim())))
        # else: already [B, C] or [B, F]
        return x

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Option A path: φ provided by connector (from PPO encoder)
        if "phi_t" in batch and "phi_tp1" in batch:
            phi      = self._project_to_feature(batch["phi_t"])
            next_phi = self._project_to_feature(batch["phi_tp1"])
        else:
            # Fallback: encode obs via ICM feature_net (if you still keep it)
            cat = torch.cat([batch[Columns.OBS], batch[Columns.NEXT_OBS]], dim=0)
            cat = self._project_to_feature(cat)
            phi, next_phi = torch.chunk(cat, 2, dim=0)

        # Actions -> normalized features
        actions = batch[Columns.ACTIONS].to(phi.device)
        a_feat  = _action_features(actions, self.action_space).view(actions.size(0), -1)

        # Sanity: must match forward_net input
        f_in = torch.cat([phi, a_feat], dim=-1)   # [B, feature_dim + a_dim]
        assert f_in.size(-1) == self._feature_dim + self._a_feat_dim, \
            f"Got {f_in.size(-1)} but expected {self._feature_dim + self._a_feat_dim}"

        # Forward loss / intrinsic reward
        pred_next_phi = self._forward_net(f_in)                      # [B, feature_dim]
        intrinsic = 0.5 * (pred_next_phi - next_phi).pow(2).sum(-1)  # [B]

        # Inverse loss head
        inv_in  = torch.cat([phi, next_phi], dim=-1)                 # [B, 2*feature_dim]
        inv_out = self._inverse_net(inv_in)                          # [B, a_dim]

        return {
            Columns.INTRINSIC_REWARDS: intrinsic,
            "pred_next_phi": pred_next_phi,
            "inverse_out": inv_out,
            "phi": phi,
            "next_phi": next_phi,
        }

    @override(SelfSupervisedLossAPI)
    def compute_self_supervised_loss(
        self,
        *,
        learner: "TorchLearner",
        module_id: ModuleID,
        config: "AlgorithmConfig",
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        module = learner.module[module_id].unwrapped()

        # Forward net loss.
        forward_loss = torch.mean(fwd_out[Columns.INTRINSIC_REWARDS])

        # Inverse loss: Box -> MSE to normalized actions; Discrete -> CE
        act_space = learner.module[module_id].unwrapped().action_space
        if isinstance(act_space, gym.spaces.Box):
            # normalize actions exactly like in _action_features
            actions = batch[Columns.ACTIONS]
            a_norm = module._action_features(actions, act_space)  # [B, 3]
            inv_targets = a_norm
            inv_preds   = fwd_out["inverse_out"]
            inverse_loss = torch.mean((inv_preds - inv_targets) ** 2)
        else:
            # Discrete/MultiDiscrete path via action distribution
            dist_inputs = module._inverse_net(torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1))
            action_dist = module.get_train_action_dist_cls().from_logits(dist_inputs)
            inverse_loss = -action_dist.logp(batch[Columns.ACTIONS]).mean()

        # Calculate the ICM loss.
        total_loss = (
            config.learner_config_dict["forward_loss_weight"] * forward_loss
            + (1.0 - config.learner_config_dict["forward_loss_weight"]) * inverse_loss
        )

        learner.metrics.log_dict(
            {
                "mean_intrinsic_rewards": forward_loss,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
            },
            key=module_id,
            window=1,
        )

        return total_loss
    def _ensure_phi_align(self, in_dim: int, device: torch.device) -> None:
        """Make (or remake) a linear projector from encoder dim -> feature_dim."""
        fd = self.model_config.get("feature_dim", 128)
        if not hasattr(self, "_phi_align_in") or self._phi_align_in != in_dim:
            self._phi_align = nn.Linear(in_dim, fd).to(device)
            self._phi_align_in = in_dim
            self._feature_dim = fd  # keep around

    def _align_phi(self, x: torch.Tensor) -> torch.Tensor:
        return self._phi_align(x)

    # Inference and exploration not supported (this is a world-model that should only
    # be used for training).
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError(
            "`IntrinsicCuriosityModel` should only be used for training! "
            "Only calls to `forward_train()` supported."
        )
    def _ensure_phi_projector(self, in_dim: int, device):
        # (Re)create if first time or input width changed
        need_new = (
            self._phi_proj is None
            or self._phi_proj.in_features != in_dim
            or self._phi_proj.out_features != self._feature_dim
        )
        if need_new:
            self._phi_proj = nn.Linear(in_dim, self._feature_dim, bias=True).to(device)

    def _to_channels_only(self, x: torch.Tensor) -> torch.Tensor:
        # Pool/flatten to [B, C] from any shape
        if x.dim() == 4:
            # Heuristic NHWC->NCHW if needed
            if x.shape[-1] in {1,3,4,8,16,32,64,128,256,512,1024} and x.shape[1] not in {1,3,4,8,16,32,64,128,256,512,1024}:
                x = x.permute(0, 3, 1, 2).contiguous()
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.dim() == 3:
            x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        elif x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim()))).view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
        return x

    def _project_to_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_channels_only(x)                   # -> [B, C*]
        self._ensure_phi_projector(x.size(-1), x.device)
        x = self._phi_proj(x)                           # -> [B, feature_dim]
        return x
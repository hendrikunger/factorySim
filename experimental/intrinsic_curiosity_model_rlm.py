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
        feature_dim = int(cfg.get("feature_dim", 128))

        # We reuse PPO’s encoder (Option A) -> keep as Identity
        self._feature_net = nn.Identity()

        # Inverse model: [phi, next_phi] -> action
        inv_in   = feature_dim * 2                     # 256
        inv_out  = ( self.action_space.n
                     if hasattr(self.action_space, "n") else
                     self._get_action_feat_dim(self.action_space) )
        self._inverse_net = self._mlp(
            inv_in, inv_out,
            cfg.get("inverse_net_hiddens", (256, 256)),
            cfg.get("inverse_net_activation", "relu"),
        )

        # Forward model: [phi, action_feat] -> next_phi
        a_feat_dim = self._get_action_feat_dim(self.action_space)  # 3 for Box(3,)
        self._forward_net = self._mlp(
            feature_dim + a_feat_dim, feature_dim,                 # 131 -> 128
            cfg.get("forward_net_hiddens", (256, 256)),
            cfg.get("forward_net_activation", "relu"),
        )



    def _flatten2(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1) if x.dim() > 2 else x
    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        if "phi_t" in batch and "phi_tp1" in batch:
            phi, next_phi = batch["phi_t"], batch["phi_tp1"]
            # ensure 2-D
            phi      = self._flatten2(phi)
            next_phi = self._flatten2(next_phi)
            # if you have a projector to feature_dim, do it AFTER flatten:
            if hasattr(self, "_ensure_phi_align"):
                self._ensure_phi_align(phi.size(-1), phi.device)
                phi      = self._align_phi(phi)
                next_phi = self._align_phi(next_phi)
        else:
            # (fallback path if ever used)
            cat = torch.cat([batch[Columns.OBS], batch[Columns.NEXT_OBS]], dim=0)
            cat_phi = self._feature_net(cat)
            phi, next_phi = torch.chunk(cat_phi, 2, dim=0)
            phi, next_phi = self._flatten2(phi), self._flatten2(next_phi)

        actions = batch[Columns.ACTIONS].to(phi.device)
        a_feat  = self._action_features(actions, self.action_space)   # [B, A]
        a_feat  = a_feat.view(a_feat.size(0), -1)                     # just in case

        pred_next_phi = self._forward_net(torch.cat([phi, a_feat], dim=-1))
        intrinsic = 0.5 * torch.sum((pred_next_phi - next_phi) ** 2, dim=-1)
        inv_out = self._inverse_net(torch.cat([phi, next_phi], dim=-1))
        return {Columns.INTRINSIC_REWARDS: intrinsic,
                "pred_next_phi": pred_next_phi,
                "inverse_out": inv_out}


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

        # Inverse loss term (predicted action that led from phi to phi' vs
        # actual action taken).
        dist_inputs = module._inverse_net(
            torch.cat([fwd_out["phi"], fwd_out["next_phi"]], dim=-1)
        )
        action_dist = module.get_train_action_dist_cls().from_logits(dist_inputs)

        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        inverse_loss = -action_dist.logp(batch[Columns.ACTIONS])
        inverse_loss = torch.mean(inverse_loss)

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

    # Inference and exploration not supported (this is a world-model that should only
    # be used for training).
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        raise NotImplementedError(
            "`IntrinsicCuriosityModel` should only be used for training! "
            "Only calls to `forward_train()` supported."
        )

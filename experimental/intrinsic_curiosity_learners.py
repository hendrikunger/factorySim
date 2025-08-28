from typing import Any, List, Optional

import gymnasium as gym
import torch
import torch.nn.functional as F
import tree  # pip install dm_tree
from ray.rllib.algorithms.dqn.torch.dqn_torch_learner import DQNTorchLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import (
    AddObservationsFromEpisodesToBatch,
)
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.connectors.learner.add_next_observations_from_episodes_to_train_batch import (  # noqa
    AddNextObservationsFromEpisodesToTrainBatch,
)
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import Columns, DEFAULT_MODULE_ID
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.typing import EpisodeType
from ray.rllib.core.models.base import ENCODER_OUT

ICM_MODULE_ID = "_intrinsic_curiosity_model"


class DQNTorchLearnerWithCuriosity(DQNTorchLearner):
    def build(self) -> None:
        super().build()
        add_intrinsic_curiosity_connectors(self)


class PPOTorchLearnerWithCuriosity(PPOTorchLearner):
    def build(self) -> None:
        super().build()
        add_intrinsic_curiosity_connectors(self)


def add_intrinsic_curiosity_connectors(torch_learner: TorchLearner) -> None:
    """Adds two connector pieces to the Learner pipeline, needed for ICM training.

    - The `AddNextObservationsFromEpisodesToTrainBatch` connector makes sure the train
    batch contains the NEXT_OBS for ICM's forward- and inverse dynamics net training.
    - The `IntrinsicCuriosityModelConnector` piece computes intrinsic rewards from the
    ICM and adds the results to the extrinsic reward of the main module's train batch.

    Args:
        torch_learner: The TorchLearner, to whose Learner pipeline the two ICM connector
            pieces should be added.
    """
    learner_config_dict = torch_learner.config.learner_config_dict

    # Assert, we are only training one policy (RLModule) and we have the ICM
    # in our MultiRLModule.
    assert (
        len(torch_learner.module) == 2
        and DEFAULT_MODULE_ID in torch_learner.module
        and ICM_MODULE_ID in torch_learner.module
    )

    # Make sure both curiosity loss settings are explicitly set in the
    # `learner_config_dict`.
    if (
        "forward_loss_weight" not in learner_config_dict
        or "intrinsic_reward_coeff" not in learner_config_dict
    ):
        raise KeyError(
            "When using the IntrinsicCuriosityTorchLearner, both `forward_loss_weight` "
            " and `intrinsic_reward_coeff` must be part of your config's "
            "`learner_config_dict`! Add these values through: `config.training("
            "learner_config_dict={'forward_loss_weight': .., 'intrinsic_reward_coeff': "
            "..})`."
        )

    if torch_learner.config.add_default_connectors_to_learner_pipeline:
        # Prepend a "add-NEXT_OBS-from-episodes-to-train-batch" connector piece
        # (right after the corresponding "add-OBS-..." default piece).
        torch_learner._learner_connector.insert_after(
            AddObservationsFromEpisodesToBatch,
            AddNextObservationsFromEpisodesToTrainBatch(),
        )
        # Append the ICM connector, computing intrinsic rewards and adding these to
        # the main model's extrinsic rewards.
        torch_learner._learner_connector.insert_after(
            NumpyToTensor,
            IntrinsicCuriosityModelConnector(
                intrinsic_reward_coeff=(
                    torch_learner.config.learner_config_dict["intrinsic_reward_coeff"]
                )
            ),
        )

def _pool_to_channels_only(x: torch.Tensor) -> torch.Tensor:
    """Return [N, C] regardless of input rank."""
    if x.dim() == 4:           # [N, C, H, W] or [N, H, W, C] (handled earlier)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    elif x.dim() == 3:         # [N, C, L]
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    elif x.dim() > 2:          # any higher rank: mean over non-batch dims
        x = x.mean(dim=tuple(range(2, x.dim()))).view(x.size(0), -1)
    else:                      # [N, F] already
        x = x.view(x.size(0), -1)
    return x

def _maybe_nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """Heuristic NHWC→NCHW for 4D tensors."""
    if x.dim() == 4:
        channelish = {1, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
        if (x.shape[-1] in channelish) and (x.shape[1] not in channelish):
            x = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    return x

class IntrinsicCuriosityModelConnector(ConnectorV2):
    """Learner ConnectorV2 piece to compute intrinsic rewards based on an ICM.

    For more details, see here:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    This connector piece:
    - requires two RLModules to be present in the MultiRLModule:
    DEFAULT_MODULE_ID (the policy model to be trained) and ICM_MODULE_ID (the instrinsic
    curiosity architecture).
    - must be located toward the end of to your Learner pipeline (after the
    `NumpyToTensor` piece) in order to perform a forward pass on the ICM model with the
    readily compiled batch and a following forward-loss computation to get the intrinsi
    rewards.
    - these intrinsic rewards will then be added to the (extrinsic) rewards in the main
    model's train batch.
    """

    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        intrinsic_reward_coeff: float,
        **kwargs,
    ):
        """Initializes a CountBasedCuriosity instance.

        Args:
            intrinsic_reward_coeff: The weight with which to multiply the intrinsic
                reward before adding it to the extrinsic rewards of the main model.
        """
        super().__init__(input_observation_space, input_action_space)

        self.intrinsic_reward_coeff = intrinsic_reward_coeff

    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Any,
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # Assert that the batch is ready.
        assert DEFAULT_MODULE_ID in batch and ICM_MODULE_ID not in batch
        assert (
            Columns.OBS in batch[DEFAULT_MODULE_ID]
            and Columns.NEXT_OBS in batch[DEFAULT_MODULE_ID]
        )

        m = rl_module[DEFAULT_MODULE_ID]
        obs      = batch[DEFAULT_MODULE_ID][Columns.OBS]
        next_obs = batch[DEFAULT_MODULE_ID][Columns.NEXT_OBS]
        actions  = batch[DEFAULT_MODULE_ID][Columns.ACTIONS]

        # --- Single encoder pass (concat then split), using PPO's encoder ---
        with torch.no_grad():
            if hasattr(m, "encoder"):
                enc = m.encoder
            elif hasattr(m, "_encoder"):
                enc = m._encoder
            else:
                # If you have a custom module, replace these with your encoder call.
                raise AttributeError("PPO RLModule has no .encoder/_encoder attribute. Expose your encoder.")

            cat_obs = tree.map_structure(
                lambda o, no: torch.cat([o, no], dim=0),
                obs,
                next_obs,
            )

            # Call RLlib encoder with a dict input and read its standard output key
            enc_in  = {Columns.OBS: cat_obs}
            enc_out = enc(enc_in) 
            cat_phi = self.extract_latent(enc_out)

            # Split back into t and t+1
            phi_t, phi_tp1 = torch.chunk(cat_phi, 2, dim=0)

            # Then call ICM with latents (as before)
            icm_in = {"phi_t": phi_t, "phi_tp1": phi_tp1, Columns.ACTIONS: actions}
            fwd_out = rl_module[ICM_MODULE_ID].forward_train(icm_in)

        # Add the intrinsic rewards to the main module's extrinsic rewards.
        batch[DEFAULT_MODULE_ID][Columns.REWARDS] += (
            self.intrinsic_reward_coeff * fwd_out[Columns.INTRINSIC_REWARDS]
        )

        # Duplicate the batch such that the ICM also has data to learn on.
        batch[ICM_MODULE_ID] = batch[DEFAULT_MODULE_ID]

        return batch
    
    def extract_latent(self, enc_out):
            """Make a single [N, F_total] latent from arbitrary ENCODER_OUT."""
            x = enc_out.get(ENCODER_OUT, enc_out) if isinstance(enc_out, dict) else enc_out

            def tensor_leaves(obj):
                if torch.is_tensor(obj):
                    return [obj]
                elif isinstance(obj, (list, tuple)):
                    out = []
                    for it in obj:
                        out.extend(tensor_leaves(it))
                    return out
                elif isinstance(obj, dict):
                    out = []
                    for v in obj.values():
                        out.extend(tensor_leaves(v))
                    return out
                return []

            leaves = tensor_leaves(x)
            if not leaves:
                raise TypeError(f"ENCODER_OUT contained no tensors; got type={type(x)}")

            # Normalize each leaf to [N, F_i], then concat along feature dim.
            feats = []
            N0 = leaves[0].size(0)
            for t in leaves:
                if not torch.is_tensor(t):
                    continue
                # Basic batch dim sanity
                if t.size(0) != N0:
                    # Try to squeeze singleton front-dims, else raise
                    t = t.view(N0, *t.shape[1:]) if t.numel() % N0 == 0 else t
                t = _maybe_nhwc_to_nchw(t)
                t = _pool_to_channels_only(t)  # -> [N, F_i]
                feats.append(t)

            if len(feats) == 1:
                return feats[0]                 # [N, F]
            return torch.cat(feats, dim=1)      # [N, ΣF_i]
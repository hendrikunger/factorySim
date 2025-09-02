from dataclasses import asdict
import gymnasium as gym
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import CNNEncoderConfig

class VisionSACCatalog(SACCatalog):
    """SAC Catalog that uses CNN encoders for image observations (H, W, C)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalize DefaultModelConfig/dataclass -> dict for safe key access.
        if not isinstance(self.model_config, dict):
            try:
                self.model_config = self.model_config.to_dict()
            except AttributeError:
                self.model_config = asdict(self.model_config)

        # Ensure SAC-required flag exists (fixes KeyError: 'twin_q').
        self.model_config.setdefault("twin_q", True)

    def _is_image_obs(self) -> bool:
        return isinstance(self.obs_space, gym.spaces.Box) and len(self.obs_space.shape) == 3

    def build_qf_encoder(self, framework: str):
        # Use a CNN for critics when obs is (H, W, C). Works with C=5 just fine.
        if self._is_image_obs():
            return CNNEncoderConfig(
                conv_filters=self.model_config.get("conv_filters"),
                activation=self.model_config.get("conv_activation", "relu"),
                # If your RLlib version supports it, you can add:
                # channels_last=True,
            ).build(framework=framework, catalog=self)
        return super().build_qf_encoder(framework)

    def build_pi_encoder(self, framework: str):
        # Same CNN for the policy encoder.
        if self._is_image_obs():
            return CNNEncoderConfig(
                conv_filters=self.model_config.get("conv_filters"),
                activation=self.model_config.get("conv_activation", "relu"),
                # channels_last=True,
            ).build(framework=framework, catalog=self)
        return super().build_pi_encoder(framework)
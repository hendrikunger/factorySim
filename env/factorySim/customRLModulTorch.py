
from ray.rllib.core.rl_module.rl_module import  RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.models.configs import ActorCriticEncoderConfig
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.configs import MLPHeadConfig

torch, nn = try_import_torch()




class MyPPOTorchRLModule(PPOTorchRLModule):
    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):

        #RLModuleConfig(observation_space=Box(0, 255, (84, 84, 2), uint8), action_space=Box(-1.0, 1.0, (3,), float64), 
        # model_config_dict={'model': 'davit_base', 'pretrained': False}, catalog_class=<class 'ray.rllib.algorithms.ppo.ppo_catalog.PPOCatalog'>
        modelconfig = self.config.model_config_dict
        timm_config = TimmEncoderConfig(input_dims = self.config.observation_space.shape[2])


        # Since we want to use PPO, which is an actor-critic algorithm, we need to
        # use an ActorCriticEncoderConfig to wrap the base encoder config.
        actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=timm_config
        )

        self.encoder = actor_critic_encoder_config.build(framework="torch")
        timm_output_dims = [timm_config.output_dims]
        #print(f"----------------------------------------->setup: {timm_output_dims}, {timm_config.model}")

        pi_config = MLPHeadConfig(
            input_dims=timm_output_dims,
            output_layer_dim=2,
        )

        vf_config = MLPHeadConfig(
            input_dims=timm_output_dims, output_layer_dim=1
        )

        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")

        self.action_dist_cls = TorchCategorical

    

class TimmEncoderConfig(ModelConfig):
    # MobileNet v2 has a flat output with a length of 1000.
    input_dims = 2
    pretrained = False
    output_dims = 1000
    model = "resnet34"

    def build(self, framework):
        assert framework == "torch", "Unsupported framework `{}`!".format(framework)
        return TimmEncoder(self)


class TimmEncoder(TorchModel, Encoder):
    """A Timm Model loader for encoders for RLlib."""

    ##
    ##Main Problem is, that the model creates a 1000 dimensional output, we need (batch, 1000)



    def __init__(self, config):
        super().__init__(config)
        self.net = timm.create_model(config.model, num_classes=config.output_dims, pretrained=config.pretrained, in_chans=config.input_dims)


    def _forward(self, input_dict, **kwargs):
        t_in = input_dict["obs"].permute(0, 3, 1, 2).float()
        print(f"Raw----------------------------- {input_dict['obs'].shape}")
        print(f"---------------------------------- {input_dict['obs'].max()}")
        print(f"---------------------------------- {input_dict['obs'].dtype}")
        print(f"Input--------------------------- ")
        print(f"---------------------------------- {t_in.shape}")
        print(f"---------------------------------- {t_in.max()}")
        print(f"---------------------------------- {t_in.dtype}")            
        forward = self.net.forward(t_in)
        print(f"Output------------------- ")
        print(f"---------------------------------- {forward.shape}")
        print(f"---------------------------------- {forward.dtype}")
        #print(f"---------------------------------- {forward}")
        return {ENCODER_OUT: (forward)}
    

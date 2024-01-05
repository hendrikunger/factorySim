from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
import timm 

from  memory_profiler import profile

torch, nn = try_import_torch()


class MyXceptionModel(TorchModelV2, nn.Module):
    """Implementation of Xception model from https://arxiv.org/abs/1610.02357."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyXceptionModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        #Xception Model
        self.model = timm.create_model('efficientnetv2_s', num_classes=num_outputs, pretrained=False, in_chans=2)
        self._value_branch = SlimFC(
                num_outputs, 1, initializer=normc_initializer(0.01), activation_fn=None
            )


    def forward(self, input_dict, state, seq_lens):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self.model(self._features)
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out

        return conv_out, state

    def value_function(self):
        assert self._features is not None, "must call forward() first"

        return self._value_branch(self._features).squeeze(1)
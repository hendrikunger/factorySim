from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.tf.tf_distributions import TfCategorical
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from typing import Any, Mapping

"""Implementation of Xception model from https://arxiv.org/abs/1610.02357."""
class MyXceptionRLModule(TfRLModule):


    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        

       #Xception Model
        img_input = tf.keras.layers.Input(shape=self.config.observation_space, name="observations")
        #input ist width, heigth, channels -> last axis as channel_axis
        channel_axis = -1
        x = tf.keras.applications.xception.preprocess_input(img_input)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)

        residual = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = tf.keras.layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
            x = tf.keras.layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = tf.keras.layers.BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

            x = tf.keras.layers.add([x, residual])

        residual = tf.keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = tf.keras.layers.BatchNormalization(axis=channel_axis)(residual)

        x = tf.keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = tf.keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
        x = tf.keras.layers.add([x, residual])

        x = tf.keras.layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = tf.keras.layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name='block14_sepconv2_act')(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)



        layer_out = tf.keras.layers.Dense(self.config.action_space, name="layer_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)

        self.policy =  tf.keras.Model(img_input, [layer_out, value_out])




    def get_train_action_dist_cls(self):
        return TfCategorical

    def get_exploration_action_dist_cls(self):
        return TfCategorical

    def get_inference_action_dist_cls(self):
        return TfCategorical

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    def _forward_shared(self, batch: NestedDict) -> Mapping[str, Any]:
        # We can use a shared forward method because BC does not need to distinguish
        # between train, inference, and exploration.
        action_logits = self.policy(batch["obs"])
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        return {"policy": self.policy.get_weights()}

    @override(RLModule)
    def set_state(self, state: Mapping[str, Any]) -> None:
        self.policy.set_weights(state["policy"])


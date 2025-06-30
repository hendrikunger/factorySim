from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer

tf1, tf, tfv = try_import_tf()


class MyXceptionModel(TFModelV2):
    """Implementation of Xception model from https://arxiv.org/abs/1610.02357."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyXceptionModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        #Xception Model
        img_input = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
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



        layer_out = tf.keras.layers.Dense(num_outputs, name="layer_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)

        self.model = tf.keras.Model(img_input, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
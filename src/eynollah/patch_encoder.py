import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
import tensorflow as tf
from tensorflow.keras import layers

class PatchEncoder(layers.Layer):

    # 441=21*21 # 14*14 # 28*28
    def __init__(self, num_patches=441, projection_dim=64):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(self.projection_dim)
        self.position_embedding = layers.Embedding(self.num_patches, self.projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        return dict(num_patches=self.num_patches,
                    projection_dim=self.projection_dim,
                    **super().get_config())

class Patches(layers.Layer):
    def __init__(self, patch_size_x=1, patch_size_y=1):
        super().__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size_y, self.patch_size_x, 1],
            strides=[1, self.patch_size_y, self.patch_size_x, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        return dict(patch_size_x=self.patch_size_x,
                    patch_size_y=self.patch_size_y,
                    **super().get_config())

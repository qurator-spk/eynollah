from keras import layers
import tensorflow as tf


class Patches(layers.Layer):
    def __init__(self, patch_size_x, patch_size_y):#__init__(self, **kwargs):#:__init__(self, patch_size):#__init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

    def call(self, images):
        #print(tf.shape(images)[1],'images')
        #print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size_y, self.patch_size_x, 1],
            strides=[1, self.patch_size_y, self.patch_size_x, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        #patch_dims = patches.shape[-1]
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

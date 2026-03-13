import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
import tensorflow as tf
from tensorflow.keras import layers, models

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

class wrap_layout_model_resized(models.Model):
    """
    replacement for layout model using resizing to model width/height and back

    (accepts arbitrary width/height input [B, H, W, 3], returns same size segmentation [B, H, W, C])
    """
    def __init__(self, model):
        super().__init__(name=model.name + '_resized')
        self.model = model
        self.height = model.layers[-1].output_shape[1]
        self.width = model.layers[-1].output_shape[2]

    @tf.function(reduce_retracing=True,
                 #jit_compile=True, (ScaleAndTranslate is not supported by XLA)
                 input_signature=[tf.TensorSpec([1, None, None, 3],
                                                dtype=tf.float32)])
    def call(self, img, training=False):
        height = tf.shape(img)[1]
        width = tf.shape(img)[2]
        img_resized = tf.image.resize(img,
                                      (self.height, self.width),
                                      antialias=True)
        pred_resized = self.model(img_resized)
        pred = tf.image.resize(pred_resized,
                               (height, width))
        return pred

    def predict(self, x, verbose=0):
        return self(x).numpy()

class wrap_layout_model_patched(models.Model):
    """
    replacement for layout model using sliding window for patches

    (accepts arbitrary width/height input [B, H, W, 3], returns same size segmentation [B, H, W, C])
    """
    def __init__(self, model):
        super().__init__(name=model.name + '_patched')
        self.model = model
        self.height = model.layers[-1].output_shape[1]
        self.width = model.layers[-1].output_shape[2]
        self.classes = model.layers[-1].output_shape[3]
        # equivalent of marginal_of_patch_percent=0.1 ...
        self.stride_x = int(self.width * (1 - 0.1))
        self.stride_y = int(self.height * (1 - 0.1))
        offset_height = (self.height - self.stride_y) // 2
        offset_width = (self.width - self.stride_x) // 2
        window = tf.image.pad_to_bounding_box(
            tf.ones((self.stride_y, self.stride_x, 1), dtype=tf.int32),
            offset_height, offset_width,
            self.height, self.width)
        self.window = tf.expand_dims(window, axis=0)

    @tf.function(reduce_retracing=True,
                 #jit_compile=True, (ScaleAndTranslate and ExtractImagePatches not supported by XLA)
                 input_signature=[tf.TensorSpec([1, None, None, 3],
                                                dtype=tf.float32)])
    def call(self, img, training=False):
        height = tf.shape(img)[1]
        width = tf.shape(img)[2]
        if (height < self.height or
            width < self.width):
            img_resized = tf.image.resize(img,
                                          (self.height, self.width),
                                          antialias=True)
            pred_resized = self.model(img_resized)
            pred = tf.image.resize(pred_resized,
                                   (height, width))
            return pred

        img_patches = tf.image.extract_patches(
            images=img,
            sizes=[1, self.height, self.width, 1],
            strides=[1, self.stride_y, self.stride_x, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        img_patches = tf.squeeze(img_patches)
        index_shape = (-1, self.height, self.width, 2)
        input_shape = (-1, self.height, self.width, 3)
        output_shape = (-1, self.height, self.width, self.classes)
        img_patches = tf.reshape(img_patches, shape=input_shape)
        # may be too large:
        #pred_patches = self.model(img_patches)
        # so rebatch to fit in memory:
        img_patches = tf.expand_dims(img_patches, 1)
        pred_patches = tf.map_fn(self.model, img_patches,
                                 parallel_iterations=1,
                                 infer_shape=False)
        pred_patches = tf.squeeze(pred_patches, 1)
        # calculate corresponding indexes for reconstruction
        x = tf.range(width)
        y = tf.range(height)
        x, y = tf.meshgrid(x, y)
        indices = tf.stack([y, x], axis=-1)
        indices_patches = tf.image.extract_patches(
            images=tf.expand_dims(indices, axis=0),
            sizes=[1, self.height, self.width, 1],
            strides=[1, self.stride_y, self.stride_x, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        indices_patches = tf.squeeze(indices_patches)
        indices_patches = tf.reshape(indices_patches, shape=index_shape)

        # use margins for sliding window approach
        indices_patches = indices_patches * self.window

        pred = tf.scatter_nd(
            indices_patches,
            pred_patches,
            (height, width, self.classes))
        pred = tf.expand_dims(pred, axis=0)
        return pred

    def predict(self, x, verbose=0):
        return self(x).numpy()

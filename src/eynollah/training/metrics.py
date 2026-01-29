from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


def weighted_categorical_crossentropy(weights=None):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """

    def loss(y_true, y_pred):
        labels_floats = tf.cast(y_true, tf.float32)
        per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats, logits=y_pred)

        if weights is not None:
            weight_mask = tf.maximum(tf.reduce_max(tf.constant(
                np.array(weights, dtype=np.float32)[None, None, None])
                                                   * labels_floats, axis=-1), 1.0)
            per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]
        return tf.reduce_mean(per_pixel_loss)

    return loss


def image_categorical_cross_entropy(y_true, y_pred, weights=None):
    """
    :param y_true: tensor of shape (batch_size, height, width) representing the ground truth.
    :param y_pred: tensor of shape (batch_size, height, width) representing the prediction.
    :return: The mean cross-entropy on softmaxed tensors.
    """

    labels_floats = tf.cast(y_true, tf.float32)
    per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats, logits=y_pred)

    if weights is not None:
        weight_mask = tf.maximum(
            tf.reduce_max(tf.constant(
                np.array(weights, dtype=np.float32)[None, None, None])
                          * labels_floats, axis=-1), 1.0)
        per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]

    return tf.reduce_mean(per_pixel_loss)


def class_tversky(y_true, y_pred):
    smooth = 1.0  # 1.00

    y_true = K.permute_dimensions(y_true, (3, 1, 2, 0))
    y_pred = K.permute_dimensions(y_pred, (3, 1, 2, 0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos), 1)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos, 1)
    alpha = 0.2  # 0.5
    beta = 0.8
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 1.3  # 4./3.0#1.3#4.0/3.00# 0.75
    return K.sum(K.pow((1 - pt_1), gamma))


def generalized_dice_coeff2(y_true, y_pred):
    n_el = 1
    for dim in y_true.shape:
        n_el *= int(dim)
    n_cl = y_true.shape[-1]
    w = K.zeros(shape=(n_cl,))
    w = (K.sum(y_true, axis=(0, 1, 2))) / n_el
    w = 1 / (w ** 2 + 0.000001)
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, (0, 1, 2))
    numerator = K.sum(numerator)
    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, (0, 1, 2))
    denominator = K.sum(denominator)
    return 2 * numerator / denominator


def generalized_dice_coeff(y_true, y_pred):
    axes = tuple(range(1, len(y_pred.shape) - 1))
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=axes)
    w = 1 / (w ** 2 + 0.000001)
    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, axes)
    numerator = K.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, axes)
    denominator = K.sum(denominator)

    gen_dice_coef = 2 * numerator / denominator

    return gen_dice_coef


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff2(y_true, y_pred)


# TODO: document where this is from
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))

    numerator = 2. * K.sum(y_pred * y_true, axes)

    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return 1.00 - K.mean(numerator / (denominator + epsilon))  # average over classes and batch


# TODO: document where this is from
def seg_metrics(y_true, y_pred, metric_name, metric_type='standard', drop_last=True, mean_per_class=False,
                verbose=False):
    """ 
    Compute mean metrics of two segmentation masks, via Keras.
    
    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)
    
    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot 
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.
    
    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    # always assume one or more classes
    num_classes = K.shape(y_true)[-1]

    if not flag_soft:
        # get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_pred = K.one_hot(K.argmax(y_pred), num_classes)
        y_true = K.one_hot(K.argmax(y_true), num_classes)

    # if already one-hot, could have skipped above command
    # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    union = mask_sum - intersection  # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = K.cast(K.not_equal(union, 0), 'float32')

    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]

    if verbose:
        print('intersection, union')
        print(K.eval(intersection), K.eval(union))
        print(K.eval(intersection / union))

    # return mean metrics: remaining axes are (batch, classes)
    if flag_naive_mean:
        return K.mean(metric)

    # take mean only over non-absent classes
    class_count = K.sum(mask, axis=0)
    non_zero = tf.greater(class_count, 0)
    non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
    non_zero_count = tf.boolean_mask(class_count, non_zero)

    if verbose:
        print('Counts of inputs with class present, metrics for non-absent classes')
        print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))

    return K.mean(non_zero_sum / non_zero_count)


# TODO: document where this is from
# TODO: Why a different implementation than IoU from utils?
def mean_iou(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.

    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)


def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes):  # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i)  # & ~void_labels
        pred_labels = K.equal(pred_pixels, i)  # & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        ious = K.sum(inter, axis=1) / K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches))))  # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def iou_vahid(y_true, y_pred):
    nb_classes = tf.shape(y_true)[-1] + tf.to_int32(1)
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []

    for i in tf.range(nb_classes):
        tp = K.sum(tf.to_int32(K.equal(true_pixels, i) & K.equal(pred_pixels, i)))
        fp = K.sum(tf.to_int32(K.not_equal(true_pixels, i) & K.equal(pred_pixels, i)))
        fn = K.sum(tf.to_int32(K.equal(true_pixels, i) & K.not_equal(pred_pixels, i)))
        iouh = tp / (tp + fp + fn)
        iou.append(iouh)
    return K.mean(iou)


# TODO: copy from utils?
def IoU_metric(Yi, y_predi):
    #  mean Intersection over Union
    #  Mean IoU = TP/(FN + TP + FP)
    y_predi = np.argmax(y_predi, axis=3)
    y_testi = np.argmax(Yi, axis=3)
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        IoUs.append(IoU)
    return K.cast(np.mean(IoUs), dtype='float32')


def IoU_metric_keras(y_true, y_pred):
    #  mean Intersection over Union
    #  Mean IoU = TP/(FN + TP + FP)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    return IoU_metric(y_true.eval(session=sess), y_pred.eval(session=sess))


# TODO: unused, remove?
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

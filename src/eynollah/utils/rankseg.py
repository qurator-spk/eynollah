# parts of https://github.com/rankseg/rankseg/blob/v0.0.8/rankseg/_rankseg_algo.py
# as needed for post-processing in .tiling
# replacing Torch with OpenCV etc.
from __future__ import annotations

from typing import Literal
import math
from numbers import Integral, Real

import numpy as np
import cv2

SUPPORTED_PROB_DTYPES = (np.float16, np.float32, np.float64, np.float128)
_SCALED_SCORE_SMOOTH_THRESHOLD = 1e6


def rankseg_rma(
    probs: np.ndarray,
    metric: str = "dice",
    smooth: float = 0.0,
    output_mode: str = "multiclass",
    pruning_prob: float = 0.5,
    unassigned_policy: str = "max_score",
    void_index: int = 255,
    safe_screening: bool | Literal["auto"] = False,
) -> np.ndarray:
    r"""
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs : Tensor, shape (batch_size, num_class, \*image_shape)
        The estimated probability tensor. Must use a real floating-point dtype
        with finite values in the range [0, 1].

    metric : str, default='dice'
        The metric aim to optimize, either 'iou' or 'dice'.

    output_mode : {'multiclass', 'multilabel'}, default='multiclass'
        Controls overlap behavior of the predictions.
        - 'multiclass': non-overlapping; each pixel belongs to exactly one class.
        - 'multilabel': overlapping; pixels can belong to multiple classes (binary mask per class).

    smooth : float, default=0.0
        A smooth parameter in the segmentation metric. When positive, the RMA
        volume search also compares the score of an empty mask (``tau=0``).

    pruning_prob : float, default=0.5
        The threshold for pruning, if all probabilities are less than or equal to `pruning_prob`,
        we skip the class.

    unassigned_policy : {'max_score', 'void'}, default='max_score'
        Policy for pixels that are not selected as positive by any class when
        converting individual binary masks to a multiclass prediction. This
        option is only applicable when ``output_mode='multiclass'``;
        ``'void'`` is rejected for multilabel output.

        RankSEG-RMA first builds one binary mask per class. In multiclass mode,
        a pixel can then be:

        - selected by exactly one class, in which case that class is used;
        - selected by multiple classes, in which case the class with the largest
          incremental score among the classes that selected the pixel is used
          (a non-selecting class cannot win);
        - selected by no class, in which case this policy is used.

        The incremental score is the RMA objective change from assigning the
        pixel after fixing pixels selected by exactly one class; it is not the
        raw pixel probability. Equal scores select the lowest eligible class
        index.

        If 'max_score', unassigned pixels are assigned to the active class with
        the largest incremental score. If pruning removes every class in a
        sample, all classes remain eligible for this fallback so that
        multiclass output can still assign a class to every pixel.

        If 'void', unassigned pixels are assigned to `void_index`.

    void_index : int, default=255
        Label used for unassigned pixels when `unassigned_policy == 'void'`.
        A non-default value is rejected unless ``output_mode='multiclass'`` and
        ``unassigned_policy='void'``.

        The value must fit in ``torch.int64`` and lie outside the valid class
        index range ``[0, num_class)`` so abstentions remain distinguishable
        from class predictions. The default value, 255, is therefore suitable
        only when there are fewer than 256 classes.

    safe_screening : bool or {'auto'}, default=False
        Control experimental two-sided screening for Dice with ``smooth=0``.
        False (the default) retains the original path; True forces screening regardless of
        input size. 'auto' screens CPU inputs, and CUDA inputs with Triton and
        at least 1,280,000 probability values (batch * classes * spatial size).
        Other CUDA inputs use optimized full sort; other devices retain the
        original path in auto mode. The threshold is empirical, not a speed
        guarantee. No probability scan or timing is added for dispatch.
        Sort only unresolved pixels, using bounded padded groups for different
        per-class lengths. Direct argmax prefers the smallest searched volume
        only on exactly equal computed maxima, without close-score retries.
        Optional Triton kernels fuse CUDA screening statistics and scoring;
        otherwise True uses pure PyTorch. Candidate size never triggers a
        full-sort retry. Other metric/smooth combinations
        retain the original full sort. Class pruning and multiclass assignment
        rules are unchanged, but binary and multiclass masks can differ.
        This can reduce sorting memory, but is not faster for every input.

    Returns
    -------
    preds : Tensor
        If ``output_mode == 'multilabel'``, returns boolean masks of shape
        (batch_size, num_class, \*image_shape) with dtype ``torch.bool``.
        Otherwise, returns class-index maps with dtype ``torch.int64`` and
        shape (batch_size, \*image_shape).

    Notes
    -----
    This is a discrete inference-time post-processing operation. Gradient
    recording is disabled internally, even when ``probs.requires_grad`` is
    ``True``.

    References
    ----------
    :cite:p:`wang2025rankseg` Wang, Z., & Dai, B. (2025). RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation. Advances in Neural Information Processing Systems (NeurIPS 2025).
    """

    assert safe_screening == False, "not (re)implemented here"

    validate_probability_tensor(probs, check_values=False)
    batch_size, num_classes, *image_shape = probs.shape
    auto_screening = False
    use_screening = False
    prepared = None
    maximum = None
    if maximum is None:
        bounds = _validate_probability_values(probs)
        maximum = bounds[1] if bounds is not None else None
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if not isinstance(output_mode, str):
        raise TypeError("output_mode must be a string")
    if not isinstance(unassigned_policy, str):
        raise TypeError("unassigned_policy must be a string")
    if not isinstance(safe_screening, bool) and not auto_screening:
        raise TypeError("safe_screening must be a bool or 'auto'")
    smooth = validate_finite_real("smooth", smooth)
    if smooth < 0:
        raise ValueError("smooth must be greater than or equal to 0")
    pruning_prob = validate_finite_real("pruning_prob", pruning_prob)
    if not 0 <= pruning_prob <= 1:
        raise ValueError("pruning_prob must be in the range [0, 1]")
    if probs.dtype in (np.float16,):
        probs = probs.astype(np.float32)

    metric = metric.strip().lower()
    output_mode = output_mode.strip().lower()
    unassigned_policy = unassigned_policy.strip().lower()
    if metric not in ["iou", "dice"]:
        raise ValueError("metric should be iou or dice")
    if output_mode not in ["multiclass", "multilabel"]:
        raise ValueError("output_mode should be multiclass or multilabel")
    if output_mode == "multiclass" and probs.shape[1] == 1:
        raise ValueError(
            "Single-channel probabilities cannot produce a binary multiclass label map; "
            "use output_mode='multilabel' or provide background and foreground channels"
        )
    if unassigned_policy not in ["max_score", "void"]:
        raise ValueError("unassigned_policy should be max_score or void")
    void_index = validate_integral("void_index", void_index)
    if output_mode == "multilabel" and unassigned_policy != "max_score":
        raise ValueError("unassigned_policy='void' is only supported when output_mode='multiclass'")
    if void_index != 255 and (output_mode != "multiclass" or unassigned_policy != "void"):
        raise ValueError("a non-default void_index requires output_mode='multiclass' and unassigned_policy='void'")
    if output_mode == "multiclass" and unassigned_policy == "void":
        int64_info = np.iinfo(np.int64)
        if not int64_info.min <= void_index <= int64_info.max:
            raise ValueError("void_index must be representable as numpy.int64")
        if 0 <= void_index < probs.shape[1]:
            raise ValueError("void_index must lie outside the valid class index range [0, num_class)")

    def compute_opt_tau(
        metric: str,
        pb_mean: np.ndarray,
        cumsum_prob: np.ndarray,
        dim: int,
        smooth: float,
    ):
        """Compute optimal tau and cutpoint based on the selected metric."""
        taus = np.arange(1, dim + 1).reshape((1, 1, -1))
        use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD
        if metric == "dice":
            denom_offset = np.expand_dims(pb_mean, -1) + taus
            if use_scaled_scores:
                metric_values = 2.0 * cumsum_prob / (1.0 + (denom_offset + 1) / smooth)
                metric_values -= denom_offset / (1.0 + denom_offset / smooth)
            else:
                discount = denom_offset + 1.0 + smooth
                metric_values = 2.0 * cumsum_prob / discount
                metric_values += smooth / (discount - 1)
        elif metric == "iou":
            denom_offset = np.expand_dims(pb_mean, -1) - cumsum_prob + taus
            if use_scaled_scores:
                metric_values = (2.0 * cumsum_prob - pb_mean.unsqueeze(-1) - taus) / (1.0 + denom_offset / smooth)
            else:
                metric_values = (cumsum_prob + smooth) / (denom_offset + smooth)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported metric: {metric}")

        # The RMA paper considers tau in {1, ..., dim} for the unsmoothed
        # objective.  A positive smooth value also gives the empty mask a
        # nonzero score, so include its exact RMA score in the extended API.
        if smooth > 0:
            if use_scaled_scores:
                empty_values = -pb_mean / (1.0 + pb_mean / smooth)
            else:
                empty_values = smooth / (pb_mean + smooth)
        else:
            empty_values = np.zeros_like(pb_mean)
        metric_values = np.concatenate([np.expand_dims(empty_values, -1), metric_values], axis=-1)

        # Get optimal tau indices, now including tau=0.
        opt_tau = np.argmax(metric_values, axis=-1)
        # cutpoint = sorted_prob[np.arange(batch_size)[:, None], np.arange(num_class), opt_tau - 1]
        return opt_tau

    def convert_to_nonoverlap(
        overlap_preds: np.ndarray,
        probs: np.ndarray,
        metric: str,
        active_mask: np.ndarray,
        pb_mean: np.ndarray,
        smooth: float,
        pruning_prob: float,
        unassigned_policy: str,
        void_index: int,
    ) -> np.ndarray:
        batch_size, _, dim = probs.shape

        class_counts = overlap_preds.sum(axis=1)
        unassigned_mask = class_counts == 0
        single_pred_mask = class_counts == 1
        has_selected_mask = class_counts > 0
        safe_to_predict = overlap_preds & np.expand_dims(single_pred_mask, 1)

        mu = (probs * safe_to_predict).sum(axis=2, keepdims=True)
        opt_tau = _count_selected_pixels(safe_to_predict, probs.dtype)

        use_scaled_scores = smooth > _SCALED_SCORE_SMOOTH_THRESHOLD

        if metric == "dice":
            denom_offset = opt_tau + np.expand_dims(pb_mean, 2) + 1
            if smooth == 0:
                # Factor the exact objective increment rather than subtracting
                # two almost equal overall scores. Large volumes can otherwise
                # lose or reverse float32 class-score differences. Threshold
                # and scale are per class, not per pixel; keep the class scale
                # because assignment compares increments across classes.
                raw_increment_scores = probs - mu / denom_offset
                raw_increment_scores *= 2.0 / (denom_offset + 1)
            elif use_scaled_scores:
                # Maximize smooth times the exact objective increment. Scaling
                # preserves the class ordering while retaining differences
                # that would round away when smooth dominates the denominator.
                scaled_base_denom = 1.0 + (denom_offset - 1) / smooth
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset + 1) / smooth
                raw_increment_scores = 2 * (probs * scaled_denom - mu / smooth) / (
                    scaled_denom * scaled_next_denom
                ) - 1.0 / (scaled_base_denom * scaled_denom)
            else:
                denom = denom_offset + smooth
                base_denom = denom - 1
                safe_base_denom = np.where(base_denom == 0, np.ones_like(base_denom), base_denom)
                raw_increment_scores = 2 * (probs * denom - mu) / (denom * (denom + 1))
                raw_increment_scores -= smooth / (safe_base_denom * denom)
                # A positive Python float can underflow to zero in a lower-
                # precision working dtype. For an empty, certainly-empty
                # class, adding a zero-probability pixel changes Dice from 1
                # to approximately 0, so its limiting increment is -1.
                raw_increment_scores = np.where(
                    base_denom == 0,
                    np.full_like(raw_increment_scores, -1.0),
                    raw_increment_scores,
                )
        else:
            denom_offset = opt_tau + np.expand_dims(pb_mean, 2) - mu
            if use_scaled_scores:
                scaled_denom = 1.0 + denom_offset / smooth
                scaled_next_denom = 1.0 + (denom_offset - probs + 1) / smooth
                raw_increment_scores = (probs * scaled_denom - (1.0 + mu / smooth) * (1.0 - probs)) / (
                    scaled_denom * scaled_next_denom
                )
            else:
                denom = denom_offset + smooth
                # With an empty mask, smooth=0, and an identically-zero class,
                # the base IoU is 0/0. Use the metric convention 0/0 := 0 so
                # that a NaN cannot make the zero-probability class win the
                # all-classes-pruned max-score fallback.
                safe_denom = np.where(denom == 0, np.ones_like(denom), denom)
                base_scores = (mu + smooth) / safe_denom
                # (a+p)/(b+1-p) - a/b = (p-(a/b)*(1-p))/(b+1-p).
                # Avoid cancellation between overall IoU scores. At b=0 the
                # validated input is an empty zero-probability class (p=a=0),
                # so the safe base score also preserves the 0/0 := 0 rule.
                remaining_prob = 1 - probs
                raw_increment_scores = probs - base_scores * remaining_prob
                raw_increment_scores /= denom + remaining_prob
                del remaining_prob

        if unassigned_policy == "max_score":
            all_classes_pruned = ~active_mask.any(axis=1, keepdims=True)
            unassigned_eligible_classes = active_mask | all_classes_pruned
        else:
            unassigned_eligible_classes = active_mask

        # At an overlap, only classes whose binary masks selected the pixel are
        # eligible. At an unassigned pixel, max_score considers active classes,
        # or every class when pruning removed all classes from that sample.
        # Reuse the binary-mask storage on CPU to avoid allocating another
        # class-by-pixel mask; accelerators are faster with one vectorized mask.
        eligible_mask = overlap_preds
        if unassigned_policy == "max_score":
            for b in range(batch_size):
                eligible_mask[b, :, unassigned_mask[b]] = unassigned_eligible_classes[b, np.newaxis]
        eligible_increment_scores = np.where(
            eligible_mask,
            raw_increment_scores,
            float("-inf"),
        )

        # Eligibility makes this argmax the final class map for single-mask,
        # overlapping, and max-score unassigned pixels alike.
        nonoverlap_predicts = eligible_increment_scores.argmax(axis=1)
        if unassigned_policy == "void":
            nonoverlap_predicts[unassigned_mask] = void_index

        return nonoverlap_predicts

    def full_sort_masks(probs: np.ndarray, pb_mean: np.ndarray):
        """Keep the original full-sort search and mask reconstruction together."""
        batch_size, num_classes, dim = probs.shape
        top_index = np.argsort(-probs, axis=-1)
        sorted_prob = np.take_along_axis(probs, top_index, axis=-1)
        active_mask = sorted_prob[:, :, 0] > pruning_prob
        cumsum_prob = np.cumsum(sorted_prob, axis=-1)
        opt_tau = compute_opt_tau(metric, pb_mean, cumsum_prob, dim, smooth)
        overlap_preds = np.zeros((batch_size, num_classes, dim), dtype=bool)
        for b in range(batch_size):
            for c in range(num_classes):
                if not active_mask[b, c]:
                    continue
                overlap_preds[b, c, top_index[b, c, : opt_tau[b, c]]] = True
        return overlap_preds, active_mask

    probs = np.reshape(probs, probs.shape[:2] + (-1,))
    if True:
        pb_mean = probs.sum(axis=-1)
        overlap_preds, active_mask = full_sort_masks(probs, pb_mean)

    if output_mode == "multilabel":
        preds = overlap_preds.reshape(batch_size, num_classes, *image_shape)
    else:
        nonoverlap_preds = convert_to_nonoverlap(
            overlap_preds,
            probs,
            metric,
            active_mask,
            pb_mean,
            smooth,
            pruning_prob,
            unassigned_policy,
            void_index,
        )
        preds = nonoverlap_preds.reshape(batch_size, *image_shape)

    return preds if output_mode == "multilabel" else preds.astype(np.int64)


def validate_probability_tensor(probs, *, check_values=True, return_bounds=False):
    """Validate probabilities, optionally reusing the already-read value range."""
    if not isinstance(probs, np.ndarray):
        raise TypeError("probs must be a Numpy array")
    if probs.dtype not in SUPPORTED_PROB_DTYPES:
        raise TypeError("probs must have a real floating-point dtype")
    if probs.ndim < 3:
        raise ValueError("probs must have shape (batch_size, num_class, *image_shape)")
    if probs.shape[1] == 0:
        raise ValueError("probs must contain at least one class")
    if any(size == 0 for size in probs.shape[2:]):
        raise ValueError("probs spatial dimensions must be non-empty")
    if check_values:
        bounds = _validate_probability_values(probs)
        if return_bounds:
            return bounds


def _validate_probability_values(probs):
    """Validate values after structure/dtype checks; return the observed range."""
    if probs.size > 0:
        minimum = probs.min()
        maximum = probs.max()
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("probs must contain only finite values")
        if minimum < 0 or maximum > 1:
            raise ValueError("probs must be in the range [0, 1]")
        return minimum, maximum

def validate_finite_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_integral(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an int")
    return int(value)


def _count_selected_pixels(mask: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Count selected pixels without discarding the caller's working precision."""
    if dtype == np.float64:
        # Integer reduction is exact before conversion and is faster than a
        # direct float64 reduction on the supported CPU/CUDA paths.
        return mask.sum(axis=2, keepdims=True).astype(dtype)
    return mask.sum(axis=2, keepdims=True, dtype=dtype)

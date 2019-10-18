import numpy as np
from .bbox import BBoxCenterToCorner, BBoxCornerToCenter
from mxnet import nd

strides = [32, 16, 8]

def _slice(x, num_anchors, num_offsets):
    """since some stages won't see partial anchors, so we have to slice the correct targets"""
    # x with shape (B, N, A, 1 or 2)
    anchors = [0] + num_anchors.tolist()
    offsets = [0] + num_offsets.tolist()
    ret = []
    for i in range(len(num_anchors)):
        y = x[:, offsets[i]:offsets[i+1], anchors[i]:anchors[i+1], :]
        ret.append(y.reshape((0, -3, -1)))
    return nd.concat(*ret, dim=1)

def PrefetchTargetGenerator(num_class, orig_height, orig_width, anchors, offsets, gt_boxes, gt_ids, gt_mixratio=None):
    """Generating training targets that do not require network predictions.

    Parameters
    ----------
    anchors : list or tuple of mxnet.nd.NDArray
        YOLO3 anchors.
    offsets : list or tuple of mxnet.nd.NDArray
        Pre-generated x and y offsets for YOLO3.
    gt_boxes : mxnet.nd.NDArray
        Ground-truth boxes.
    gt_ids : mxnet.nd.NDArray
        Ground-truth IDs.
    gt_mixratio : mxnet.nd.NDArray, optional
        Mixup ratio from 0 to 1.

    Returns
    -------
    (tuple of) mxnet.nd.NDArray
        objectness: 0 for negative, 1 for positive, -1 for ignore.
        center_targets: regression target for center x and y.
        scale_targets: regression target for scale x and y.
        weights: element-wise gradient weights for center_targets and scale_targets.
        class_targets: a one-hot vector for classification.

    """
    assert isinstance(anchors, (list, tuple))
    all_anchors = nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
    assert isinstance(offsets, (list, tuple))
    all_offsets = nd.concat(*[o.reshape(-1, 2) for o in offsets], dim=0)
    num_anchors = np.cumsum([a.size // 2 for a in anchors])
    num_offsets = np.cumsum([o.size // 2 for o in offsets])
    _offsets = [0] + num_offsets.tolist()
    feat_shapes = [(orig_height // strides[i], orig_width // strides[i]) for i in range(3)]

    # outputs
    shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
        (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
    center_targets = nd.zeros_like(shape_like)
    scale_targets = nd.zeros_like(center_targets)
    weights = nd.zeros_like(center_targets)
    objectness = nd.zeros_like(weights.split(axis=-1, num_outputs=2)[0])
    class_targets = nd.one_hot(objectness.squeeze(axis=-1), depth=num_class)
    class_targets[:] = -1  # prefill -1 for ignores

    # for each ground-truth, find the best matching anchor within the particular grid
    # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
    # then only the anchor in (3, 4) is going to be matched
    gtx, gty, gtw, gth = BBoxCornerToCenter(gt_boxes, axis=-1, split=True)
    shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)
    anchor_boxes = nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors
    shift_anchor_boxes = BBoxCenterToCorner(anchor_boxes, axis=-1, split=False)
    ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))
    # real value is required to process, convert to Numpy
    matches = ious.argmax(axis=1).asnumpy()  # (B, M)
    valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # (B, M)
    np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
    np_anchors = all_anchors.asnumpy()
    np_gt_ids = gt_ids.asnumpy()
    np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None
    # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
    # should not be a problem right now. Switch to better solution is needed.
    for b in range(matches.shape[0]):
        for m in range(matches.shape[1]):
            if valid_gts[b, m] < 1:
                break
            match = int(matches[b, m])
            nlayer = np.nonzero(num_anchors > match)[0][0]
            height = feat_shapes[nlayer][0]
            width = feat_shapes[nlayer][1]
            gtx, gty, gtw, gth = (np_gtx[b, m, 0], np_gty[b, m, 0],
                                    np_gtw[b, m, 0], np_gth[b, m, 0])
            # compute the location of the gt centers
            loc_x = int(gtx / orig_width * width)
            loc_y = int(gty / orig_height * height)
            # write back to targets
            index = _offsets[nlayer] + loc_y * width + loc_x
            center_targets[b, index, match, 0] = gtx / orig_width * width - loc_x  # tx
            center_targets[b, index, match, 1] = gty / orig_height * height - loc_y  # ty
            scale_targets[b, index, match, 0] = np.log(gtw / np_anchors[match, 0])
            scale_targets[b, index, match, 1] = np.log(gth / np_anchors[match, 1])
            weights[b, index, match, :] = 2.0 - gtw * gth / orig_width / orig_height
            objectness[b, index, match, 0] = (
                np_gt_mixratios[b, m, 0] if np_gt_mixratios is not None else 1)
            class_targets[b, index, match, :] = 0
            class_targets[b, index, match, int(np_gt_ids[b, m, 0])] = 1
    # since some stages won't see partial anchors, so we have to slice the correct targets
    objectness = _slice(objectness, num_anchors, num_offsets)
    center_targets = _slice(center_targets, num_anchors, num_offsets)
    scale_targets = _slice(scale_targets, num_anchors, num_offsets)
    weights = _slice(weights, num_anchors, num_offsets)
    class_targets = _slice(class_targets, num_anchors, num_offsets)
    return objectness, center_targets, scale_targets, weights, class_targets

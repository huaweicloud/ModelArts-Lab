import mxnet as mx

def BBoxSplit(input_data, axis, squeeze_axis=False):
    return mx.sym.split(data=input_data, axis=axis, num_outputs=4, squeeze_axis=squeeze_axis)

def BBoxCenterToCorner(input_data, axis=-1):
    x, y, w, h = mx.sym.split(data=input_data, axis=axis, num_outputs=4)
    hw = w / 2
    hh = h / 2
    xmin = x - hw
    ymin = y - hh
    xmax = x + hw
    ymax = y + hh
    return xmin, ymin, xmax, ymax

def BBoxBatchIOU(box_preds, gt_boxes, axis=-1, fmt='corner', offset=0, eps=1e-15):
    if fmt.lower() == 'center':
        al, at, ar, ab = BBoxCenterToCorner(box_preds)
        bl, bt, br, bb = BBoxCenterToCorner(gt_boxes)
    else:
        al, at, ar, ab = BBoxSplit(box_preds, axis=axis, squeeze_axis=True)
        bl, bt, br, bb = BBoxSplit(gt_boxes, axis=axis, squeeze_axis=True)

    left = mx.sym.broadcast_maximum(
        mx.sym.expand_dims(data=al, axis=-1), mx.sym.expand_dims(data=bl, axis=-2))
    right = mx.sym.broadcast_minimum(
        mx.sym.expand_dims(data=ar, axis=-1), mx.sym.expand_dims(data=br, axis=-2))
    top = mx.sym.broadcast_maximum(
        mx.sym.expand_dims(data=at, axis=-1), mx.sym.expand_dims(data=bt, axis=-2))
    bot = mx.sym.broadcast_minimum(
        mx.sym.expand_dims(data=ab, axis=-1), mx.sym.expand_dims(data=bb, axis=-2))

    # clip with (0, float16.max)
    iw = mx.sym.clip(data=right - left + offset, a_min=0, a_max=6.55040e+04)
    ih = mx.sym.clip(data=bot - top + offset, a_min=0, a_max=6.55040e+04)
    i = iw * ih

    # areas
    area_a = mx.sym.expand_dims(data=(ar - al + offset) * (ab - at + offset), axis=-1)
    area_b = mx.sym.expand_dims(data=(br - bl + offset) * (bb - bt + offset), axis=-2)
    union = mx.sym.broadcast_add(area_a, area_b) - i

    return i / (union + eps)

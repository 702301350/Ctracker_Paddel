import paddle

def hard_nms(boxes, scores, threshold=0.7):
    keep = paddle.vision.ops.nms(boxes, scores=scores, iou_threshold=threshold)
    return keep

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import yolo_detection.core.utils as utils

from yolo_detection.core.yolov4 import YOLOv4, decode, filter_boxes

from yolo_detection.core.config import cfg


def save_tf():
    STRIDES = [8, 16, 32]
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, True)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE

    input_layer = tf.keras.layers.Input([416, 416, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    prob_tensors = []

    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(fm, 416 // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,)
        else:
            output_tensors = decode(fm, 416 // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])

    pred_bbox = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in bbox_tensors]
    pred_bbox = tf.concat(pred_bbox, axis=1)
    pred_prob = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in prob_tensors]
    pred_prob = tf.concat(pred_prob, axis=1)
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.2)
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model, 'C:/Users/user/PycharmProjects/BRAIN.net/yolo_detection/data/yolov4.weights')
    model.summary()
    model.save('data', save_format='tf')

save_tf()

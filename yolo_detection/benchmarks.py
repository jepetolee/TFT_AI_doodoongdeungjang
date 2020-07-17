import numpy as np
import tensorflow as tf
import time
import cv2
from yolo_detection.core.yolov4 import YOLOv4, decode
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from yolo_detection.core import utils
from yolo_detection.core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def main(_argv):
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm,1, NUM_CLASS,STRIDES,ANCHORS, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
     utils.load_weights(model, "C:/Users/user/PycharmProjects/BRAIN.net/yolo_detection/data/yolov4.weights")

    logging.info('weights loaded')

    @tf.function
    def run_model(x):
        return model(x)

    # Test the TensorFlow Lite model on random input data.
    sum = 0
    image = "C:/Users/user/PycharmProjects/BRAIN.net/yolo_detection/data/kite.jpg"
    size = 416
    original_image = cv2.imread(image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preprocess(np.copy(original_image), [416, 416])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    batched_input = tf.constant(image_data)
    for i in range(1000):
        prev_time = time.time()
        # pred_bbox = model.predict(image_data)

        pred_bbox = []
        result = run_model(image_data)
        for value in result:
            value = value.numpy()
            pred_bbox.append(value)

        # pred_bbox = pred_bbox.numpy()
        curr_time = time.time()
        exec_time = curr_time - prev_time
        if i == 0: continue
        sum += (1 / exec_time)
        info = str(i) + " time:" + str(round(exec_time, 3)) + " average FPS:" + str(
            round(sum / i, 2)) + ", FPS: " + str(
            round((1 / exec_time), 1))
        print(info)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for input_reader_builder."""

import os
import numpy as np
import tensorflow as tf
import sys
from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from object_detection.builders import input_reader_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
import cv2
from object_detection.core import batcher
from utils import label_map_util

sys.path.append("..")

from utils import label_map_util


## to use the code, you need change PATH_TO_CKPT, tf_record_path, RESULT_PATH, total(which is determined by tf_record_path)

from utils import visualization_utils as vis_util
#PATH_TO_CKPT = r'/your/path/to/models/object_detection/models/model/dota608_ssd608_output_1243788/frozen_inference_graph.pb'
PATH_TO_CKPT = r'/your/path/to/models/object_detection/models/model/dota_rfcn_output_2000000_136610/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'dota_label_map.pbtxt')

NUM_CLASSES = 15
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'

# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 6) ]

#testsetpath = r'/your/path/to/data/dota608/test.txt'
testsetpath = r'/your/path/to/data/dota/test.txt'

with open(testsetpath, 'r') as f:
    lines = f.readlines()
    TEST_IMAGE_PATHS = [x.strip() for x in lines]

# Size, in inches, of the output images.
IMAGE_SIZE = (20, 15)
RESULT_PATH = r'/your/path/to/models/object_detection/models/model/results/dota_rfcn_output_2000000_136610_test'
#RESULT_PATH = r'/your/path/to/models/object_detection/models/model/results/dota608_ssd608_output_1243788_test'

tf_record_path = r'/your/path/to/data/dota/tf_records/dota_test.record'
#tf_record_path = r'/your/path/to/data/dota608/tf_records/dota_test_608.record'
input_reader_text_proto = """
  shuffle: false
  num_readers: 1
  tf_record_input_reader {{
    input_path: '{0}'
  }}
""".format(tf_record_path)
print 'ffffffffffffff'

# sv = tf.train.Supervisor(logdir=self.get_temp_dir())

# sv = tf.train.Supervisor(logdir=self.get_temp_dir())

# tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
#     tensor_dict[fields.InputDataFields.image], 0)
#
# images = tensor_dict[fields.InputDataFields.image]
# float_images = tf.to_float(images)
# tensor_dict[fields.InputDataFields.image] = float_images


# input_queue = batcher.BatchQueue(
#     tensor_dict,
#     batch_size=30,
#     batch_queue_capacity=100,
#     num_batch_queue_threads=4,
#     prefetch_queue_capacity=100)

### it seems that the code must have a feed_dict, so I have to run the queue.pop(tensor_dict) to get the data then feed the feed_dict.
#  Commonly, if we have a queue, can we direct use the tensor, and run the final output tensor?
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = input_reader_builder.build(input_reader_proto)
        print 'type tensor_dict', type(tensor_dict)
        #print 'tensor_dict', tensor_dict
  # sv.start_queue_runners(sess)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        outfileset = {}
        ## for bod-v3
        total = len(TEST_IMAGE_PATHS)
	print('lmages len:', total)
        ## for dota608
        #total = 22010
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        testnames = []
        for index in range(total):
        #while not coord.should_stop():
            print str(index) + '/' + str(total)
            output_dict = sess.run(tensor_dict)
            #print 'output_dict shape', tf.shape(tensor_dict)

            image = output_dict[fields.InputDataFields.image]
            source_id = output_dict[fields.InputDataFields.source_id]
            filename = output_dict[fields.InputDataFields.filename]
            height, width, depth = image.shape
            #print 'shape:', output_dict[fields.InputDataFields.image].shape
            #print 'source_id', output_dict[fields.InputDataFields.source_id]
            #print 'filename', output_dict[fields.InputDataFields.filename]
            #print 'image:', output_dict[fields.InputDataFields.image]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #subimgname = os.path.basename(os.path.splitext(filename)[0])
            subimgname = filename
            testnames.append(subimgname)
            # print('shape boxes:', np.shape(boxes))
            # print('shape scores:', np.shape(scores))
            # print('shape classes:', np.shape(classes))
            min_score_thresh = 0.1
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            for i in range(boxes.shape[0]):
                if scores[i] > min_score_thresh:
                    #print('boxes[i]:', boxes[i])

                    for j in range(len(boxes[i])):
                        if (j % 2) == 0:
                            boxes[i][j] = boxes[i][j] * height
                        else:
                            boxes[i][j] = boxes[i][j] * width
                    box = tuple(boxes[i].tolist())
                    #box = tuple((1024 * boxes[i]).tolist())
                    #print('box:', box)
                    assert classes[i] in category_index.keys(), 'the class is not in label_map'
                    class_name = category_index[classes[i]]['name']
                    ymin, xmin, ymax, xmax = box
                    outline = subimgname + ' ' + str(scores[i]) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(
                        xmax) + ' ' + str(ymax)
                    if class_name not in outfileset:
                        outfilename = os.path.join(RESULT_PATH, 'comp4_det_test_' + class_name + '.txt')
                        outfileset[class_name] = open(outfilename, 'w')
                    outfileset[class_name].write(outline + '\n')
        coord.request_stop()
        coord.join(threads)
        testnamesset = set(testnames)
        print 'testnamesset len:', len(testnamesset)





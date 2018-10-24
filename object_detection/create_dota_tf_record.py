r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_pascal_tf_record.py --data_dir=/data/dota/train/ \
        --indexfile=train.txt
        --output_name=dota_train.record
"""

import tensorflow as tf
import utils.utils as util
import sys
import os
import io
import PIL.Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', r'', 'Path to the data set to be converted.')
flags.DEFINE_string('indexfile', 'train.txt', 'All the full path of the training set or test set.') # put it under the FLAGS.data_dir path.
flags.DEFINE_string('output_name', 'dota_train.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', r'data/dota_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS
def create_tf_example(data,
                      imagepath,
                      label_map_dict,
                      filename,
                      ignore_difficult_instances=True
                      ):
  # TODO(user): Populate the following variables from your example.

  full_path = os.path.join(imagepath, filename + '.png')
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  if image.format != 'PNG':
    raise ValueError('Image format not PNG')

  width = 1024
  height = 1024
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  difficult_obj = []
  for obj in data:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
    # if ((float(obj['bndbox'][0]) < 0) or
    #     (float(obj['bndbox'][1]) < 0) or
    #     (float(obj['bndbox'][2]) >= 1024) or
    #     (float(obj['bndbox'][3]) >= 1024) ):
    #     continue
    xmin = max(obj['bndbox'][0], 0)
    ymin = max(obj['bndbox'][1], 0)
    xmax = min(obj['bndbox'][2], width - 1)
    ymax = min(obj['bndbox'][3], height - 1)

    difficult_obj.append(int(difficult))

    xmins.append(float(xmin) / width)
    ymins.append(float(ymin) / height)
    xmaxs.append(float(xmax) / width)
    ymaxs.append(float(ymax) / height)

    # xmins.append(float(obj['bndbox'][0]) / width)
    # ymins.append(float(obj['bndbox'][1]) / height)
    # xmaxs.append(float(obj['bndbox'][2]) / width)
    # ymaxs.append(float(obj['bndbox'][3]) / height)

    classes_text.append(obj['name'].encode('utf8'))
    if (obj['name'] in label_map_dict):
        classes.append(label_map_dict[obj['name']])

    else:
        #print '>>>>>>>>>>>>>'
        continue


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  #print 'tf_example: ', tf_example
  return tf_example

def main(_):
    data_dir = FLAGS.data_dir
    indexfile = FLAGS.indexfile
    if not os.path.exists(os.path.join(data_dir, indexfile)):
        # print os.path.join(data_dir, indexfile)
        raise ValueError('{} not in the path: {}'.format(indexfile, data_dir))

    output_path = os.path.join(data_dir, 'tf_records')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, FLAGS.output_name))
    print ('start-------')
    # TODO(user): Write code to read in your dataset to examples variable

    imagepath = os.path.join(data_dir, 'images')
    f = open(os.path.join(data_dir, indexfile), 'r')
    lines = f.readlines()
    txtlist = [x.strip().replace(r'images', r'labelTxt').replace('.png', '.txt') for x in lines]
    # txtlist = util.GetFileFromThisRootDir(os.path.join(data_dir, 'wordlabel'))
    for fullname in txtlist:
        data = util.parse_bod_rec(fullname)
        # print 'len(data):', len(data)
        # print 'data:', data
        # assert len(data) >= 0, "there exists empty data: " + fullname
        basename = os.path.basename(os.path.splitext(fullname)[0])
        label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
        # print 'label_map_dict', label_map_dict
        tf_example = create_tf_example(data,
                                       imagepath,
                                       label_map_dict,
                                       basename)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
  tf.app.run()

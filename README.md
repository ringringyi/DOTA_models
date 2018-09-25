# DOTA_models

We provide the config files, TFRecord files and label_map file used in training [DOTA](http://captain.whu.edu.cn/DOTAweb/dataset.html) with ssd and rfcn, and the trained models have been uploaded to Baidu Drive.   
Notice that our code is tested on official [Tensorflow models@(commit fe2f8b01c6)](https://github.com/tensorflow/models/tree/fe2f8b01c686fd62272c3992686a637db926ce5c) with [tf-nightly-gpu (1.5.0.dev20171102)](https://pypi.org/project/tf-nightly-gpu/), cuda-8.0 and cudnn-6.0 on Ubuntu 16.04.1 LTS.

## Installation
- [Tensorflow](https://pypi.org/project/tf-nightly-gpu/):
  ```bash
      pip install tf-nightly-gpu==1.5.0.dev20171102
  ```
- [Object Detection API](https://github.com/ringringyi/DOTA_models/tree/master/object_detection)<br>
  Follow the instructions in [Installation](https://github.com/ringringyi/DOTA_models/blob/master/object_detection/g3doc/installation.md).
- [Development kit](https://github.com/CAPTAIN-WHU/DOTA_devkit)<br>
  You can easily install it following the instructions in [readme](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/readme.md).

## Preparing inputs
Tensorflow Object Detection API reads data using the TFRecord file format. The raw DOTA data set is located [here](http://captain.whu.edu.cn/DOTAweb/dataset.html). To download, extract and convert it to TFRecords, run the following commands
below:
```bash
# From tensorflow/models/object_detection/
python create_dota_tf_record.py \
    --label_map_path=data/dota_label_map.pbtxt \
    --data_dir=/path/to/dota/
```
The label map for DOTA data set can be found at `data/dota_label_map.pbtxt`.

## Training
A local training job can be run with the following command:

```bash
# From tensorflow/models/object_detection/
python train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
```
The pipline config file for DOTA data set can be found at `models/model/rfcn_resnet101_dota.config` or  `models/model/ssd608_inception_v2_dota608.config`.

Here we train rfcn with image size of 1024×1024, ssd with image size of 608×608. Please refer to [DOTA_devkit/ImgSplit.py](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/ImgSplit.py) to split the picture and label. The trained models can be downloaded here: 
Baidu Drive: [rfcn](https://pan.baidu.com/s/15fFYrffdF94UzA5tYq6ToQ), [ssd](https://pan.baidu.com/s/1Gg4KYlqBtyp83DHJW1qTxg)
Google Drive: [rfcn](https://drive.google.com/open?id=1IIyTRcV1LcCqiyU1xTWftOnOD015ka2P), [ssd](https://drive.google.com/open?id=1Kt82V0PG4hJ6rCsFDnrhAGTbOw0v7xYK)

## Evaluation
You can use the pre-trained models to test images. Modify paths in `getresultfromtfrecord.py` and then run with the following commad:
```bash
# From tensorflow/models/object_detection/
python getresultfromtfrecord.py
```
Then you will obtain 15 files in the specified folder. For DOTA, you can submit your results on [Task2 - Horizontal Evaluation Server](http://captain.whu.edu.cn/DOTAweb/evaluation.html) for evaluation. Make sure your submission is in the correct format. 

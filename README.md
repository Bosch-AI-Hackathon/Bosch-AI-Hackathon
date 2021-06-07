Delivarables:

Model Folder - deep learning model

label_map.pbtxt - labels for detection

image_eval.py - image inference script

video_eval.py - video inference script

generate_tfrecord.py - script to generate record file from pascalvoc xml label file.

For Infering Image: python image_eval.py -I path-to-image -M path-to-model -L path-to-label_map

For Infering Video: python video_eval.py -V path-to-video -M path-to-model -L path-to-label_map

For creating TF record:

Create train data: python generate_tfrecord.py -x [PATH_TO_TRAIN_IMAGES_FOLDER] -l [PATH_TO_label_map.pbtxt] -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

Create test data: python generate_tfrecord.py -x [PATH_TO_TEST_IMAGES_FOLDER] -l [PATH_TO_label_map.pbtxt] -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

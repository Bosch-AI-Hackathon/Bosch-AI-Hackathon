Delivarables:

Model Folder
label_map.pbtxt
image_eval.py
video_eval.py
generate_record.py

For Infering Image: python image_eval.py -I path-to-image -M path-to-model -L path-to-label

For Infering Video: python video_eval.py -V path-to-video -M path-to-model -L path-to-label

For creating TF record:

Create train data:
python generate_tfrecord.py -x [PATH_TO_TRAIN_IMAGES_FOLDER] -l [PATH_TO_label_map.pbtxt] -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

Create test data:
python generate_tfrecord.py -x [PATH_TO_TEST_IMAGES_FOLDER] -l [PATH_TO_label_map.pbtxt] -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record


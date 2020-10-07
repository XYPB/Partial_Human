# Input Image Directory
CUSTOM_DIR=~/umcs/partial_humans/mypic

# If on a multi-GPU machine, can set os.env['']
GPU=0

CMD="python -m src.datasets.custom_dataset_to_tfrecords
--CUSTOM_DIR=${CUSTOM_DIR} --GPU=${GPU}"

echo $CMD
$CMD

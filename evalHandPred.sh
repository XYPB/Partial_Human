#! /bin/zsh
# This script get the hand prediction result from different model

source ~/.zshrc
cd utils
jsonPth=""
dataPth=""

echo "data set $1 is evaluated"
if [[ $1 == "vlog" ]]; then
    jsonPth="../data/vlog_imgs.json"
    dataPth="../data"
elif [[ $1 == "all" ]]; then
    jsonPth="../data/vlog_imgs.json"
    dataPth="../frame_cache"
fi

if [[ $jsonPth == "" || $dataPth == "" ]]; then
    exit(1)
fi

if [[ $? == 0 ]]; then
    python gen_json.py
fi

cd ../hmr && conda activate hmr
if [[ $? == 0 ]]; then
    echo "activating hmr"
    python -m demo --img_path $jsonPth --data_path $dataPth
fi

cd ../hand_detector.d2 && conda activate 100doh
if [[ $? == 0 ]]; then
    echo "activating hand detector"
    python demo.py --json_path=$jsonPth --data_path=$dataPth
fi

cd ../hand_object_detector && conda activate handobj
if [[ $? == 0 ]]; then
    echo "activating hand object detector"
    python demo.py --cuda --checkepoch=8 --checkpoint=132028 --json_dir=$jsonPth --image_dir=$dataPth
fi
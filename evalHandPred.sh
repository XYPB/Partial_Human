#! /bin/bash
# This script get the hand prediction result from different model

GPU=0

source ~/.bashrc
cd utils
if [[ "$(echo $?)" == 0 ]]; then
    echo "python gen_json"
    python gen_json.py
fi

cd ../hmr && conda activate hmr
if [[ $? == 0 ]]; then
    python -m demo
fi

cd ../hand_detector.d2 && conda activate 100doh
if [[ $? == 0 ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python demo.py
fi

cd ../hand_object_detector && conda activate handobj
if [[ $? == 0 ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python demo.py --cuda --checkepoch=8 --checkpoint=132028
fi
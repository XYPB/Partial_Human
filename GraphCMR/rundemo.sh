#! /bin/zsh
for i in ~/umcs/partial_humans/mypic/images/*; do
    if echo $i | grep preds; then
        continue
    fi
    cmd="python demo.py --checkpoint=data/models/ours/2020_02_29-18_30_01.pt --img $i"
    echo $cmd
    python demo.py --checkpoint=data/models/ours/2020_02_29-18_30_01.pt --img $i
done
#! /bin/zsh
for i in ~/umcs/partial_humans/mypic/images/*; do
    if echo $i | grep pred; then
        continue
    fi
    cmd="python -m demo --img_path \"$i\""
    echo $cmd
    python -m demo --img_path $i
done
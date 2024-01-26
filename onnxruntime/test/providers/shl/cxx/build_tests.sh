#!/bin/bash

run_make() {
    dir=$1
    cmd=$2
    (cd "$dir" && make $cmd)
}

directories=("mobilenetv2" "resnet50" "retinaface" "yolov5n")

if [ "$1" = "clean" ]; then
    for dir in "${directories[@]}"; do
        run_make "$dir" clean
    done
else
    for dir in "${directories[@]}"; do
        run_make "$dir"
    done
fi

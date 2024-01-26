#!/bin/sh

run_test() {
    dir=$1
    (cd "$dir" && ./test_$dir.sh)
}

directories=("mobilenetv2" "resnet50" "retinaface" "yolov5n")

for dir in "${directories[@]}"; do
    run_test "$dir"
done

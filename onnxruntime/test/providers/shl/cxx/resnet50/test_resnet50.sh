./run_resnet50 && ./run_resnet50 --use_shl && ./cosine_similarity shl_resnet50_out0.bin cpu_resnet50_out0.bin 0.9
rm -rf shl_resnet50_out0.bin cpu_resnet50_out0.bin

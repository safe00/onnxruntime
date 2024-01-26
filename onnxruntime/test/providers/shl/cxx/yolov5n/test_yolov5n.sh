./run_yolov5n  \
&& ./run_yolov5n --use_shl  \
&& ./cosine_similarity shl_yolov5n_out0.bin cpu_yolov5n_out0.bin 0.9  \
&& ./cosine_similarity shl_yolov5n_out1.bin cpu_yolov5n_out1.bin 0.9  \
&& ./cosine_similarity shl_yolov5n_out2.bin cpu_yolov5n_out2.bin 0.9  \
&& rm -rf shl_yolov5n_out0.bin cpu_yolov5n_out0.bin shl_yolov5n_out1.bin cpu_yolov5n_out1.bin shl_yolov5n_out2.bin cpu_yolov5n_out2.bin

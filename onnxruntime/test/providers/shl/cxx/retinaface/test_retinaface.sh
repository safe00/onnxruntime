./run_retinaface  \
&& ./run_retinaface --use_shl  \
&& ./cosine_similarity shl_retinaface_out0.bin cpu_retinaface_out0.bin 0.9  \
&& ./cosine_similarity shl_retinaface_out1.bin cpu_retinaface_out1.bin 0.9  \
&& ./cosine_similarity shl_retinaface_out2.bin cpu_retinaface_out2.bin 0.9  \
&& rm -rf shl_retinaface_out0.bin cpu_retinaface_out0.bin shl_retinaface_out1.bin cpu_retinaface_out1.bin shl_retinaface_out2.bin cpu_retinaface_out2.bin

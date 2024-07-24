# DeepSpeed-for-Hybrid-GPU
本课题基于DeepSpeedv0.9.5，修改其在张量并行推理时的张量切分方法，根据GPU的算力或显存分配对应的张量，将更多的张量分配给算力或显存高的GPU，使其能够在混合GPU的硬件环境下对推理进行加速。

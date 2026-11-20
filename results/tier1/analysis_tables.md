### 表 T1：纯 NCCL all-reduce 的形状敏感性（不加载模型）

| 运行 | 测试 | 数据类型 | 比较对象 | 位不同的元素 | 最大差 | 位级一致 |
|---|---|---|---|---|---|---|
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | bfloat16 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | bfloat16 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | bfloat16 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | bfloat16 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | bfloat16 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | float32 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | float32 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | float32 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | float32 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | reshape | float32 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 63 → 64 | 0 / 129024 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 64 → 65 | 0 / 131072 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 127 → 128 | 0 / 260096 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 128 → 129 | 0 / 262144 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 255 → 256 | 0 / 522240 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 256 → 257 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | bfloat16 | 511 → 512 | 0 / 1046528 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 63 → 64 | 0 / 129024 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 64 → 65 | 0 / 131072 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 127 → 128 | 0 / 260096 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 128 → 129 | 0 / 262144 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 255 → 256 | 0 / 522240 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 256 → 257 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | pad | float32 | 511 → 512 | 0 / 1046528 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | bfloat16 | exact_fp64 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | bfloat16 | rank_order | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | bfloat16 | reverse_order | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | float32 | exact_fp64 | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | float32 | rank_order | 0 / 524288 | 0.0 | ✅ |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | order | float32 | reverse_order | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | bfloat16 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | bfloat16 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | bfloat16 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | bfloat16 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | bfloat16 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | float32 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | float32 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | float32 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | float32 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | reshape | float32 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 63 → 64 | 1016 / 129024 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 64 → 65 | 31612 / 131072 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 127 → 128 | 1072 / 260096 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 128 → 129 | 35121 / 262144 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 255 → 256 | 1061 / 522240 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 256 → 257 | 34940 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | bfloat16 | 511 → 512 | 1116 / 1046528 | 1.562e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 63 → 64 | 1048 / 129024 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 64 → 65 | 18442 / 131072 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 127 → 128 | 1139 / 260096 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 128 → 129 | 18522 / 262144 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 255 → 256 | 1019 / 522240 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 256 → 257 | 18528 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | pad | float32 | 511 → 512 | 1127 / 1046528 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | bfloat16 | exact_fp64 | 144667 / 524288 | 1.562e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | bfloat16 | rank_order | 148104 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | bfloat16 | reverse_order | 161481 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | float32 | exact_fp64 | 148440 / 524288 | 2.384e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | float32 | rank_order | 145974 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | order | float32 | reverse_order | 159632 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | bfloat16 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | bfloat16 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | bfloat16 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | bfloat16 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | bfloat16 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | float32 | 256 → 2048 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | float32 | 128 → 4096 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | float32 | 512 → 1024 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | float32 | 1024 → 512 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | reshape | float32 | 2048 → 256 | 0 / 524288 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 63 → 64 | 0 / 129024 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 64 → 65 | 10071 / 131072 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 127 → 128 | 0 / 260096 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 128 → 129 | 10065 / 262144 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 255 → 256 | 0 / 522240 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 256 → 257 | 10252 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | bfloat16 | 511 → 512 | 0 / 1046528 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 63 → 64 | 0 / 129024 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 64 → 65 | 4861 / 131072 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 127 → 128 | 0 / 260096 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 128 → 129 | 4791 / 262144 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 255 → 256 | 0 / 522240 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 256 → 257 | 4912 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | pad | float32 | 511 → 512 | 0 / 1046528 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | bfloat16 | exact_fp64 | 157281 / 524288 | 1.562e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | bfloat16 | rank_order | 167354 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | bfloat16 | reverse_order | 108024 / 524288 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | float32 | exact_fp64 | 157739 / 524288 | 2.384e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | float32 | rank_order | 167574 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | order | float32 | reverse_order | 104535 / 524288 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 2 | 607 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 4 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 8 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 16 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 32 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 64 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 96 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 128 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 160 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 192 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 224 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 240 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 255 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 256 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 257 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 288 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | bfloat16 | 1 → 300 | 593 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 2 | 545 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 4 | 545 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 8 | 545 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 16 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 32 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 64 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 96 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 128 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 160 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 192 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 224 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 240 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 255 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 256 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 257 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 288 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse | float32 | 1 → 300 | 546 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 2 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 4 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 8 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 16 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 32 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 64 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 96 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 128 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 160 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 192 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 224 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 240 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 255 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 256 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 257 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 288 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | bfloat16 | 1 → 300 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 2 | 0 / 2048 | 0.0 | ✅ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 4 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 8 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 16 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 32 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 64 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 96 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 128 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 160 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 192 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 224 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 240 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 255 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 256 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 257 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 288 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | sparse | float32 | 1 → 300 | 349 / 2048 | 4.768e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 2 | 448 / 2048 | 6.250e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 8 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 32 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 64 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 128 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 192 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 240 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 255 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 256 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 257 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | bfloat16 | 1 → 300 | 447 / 2048 | 3.125e-02 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 2 | 460 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 8 | 460 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 32 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 64 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 128 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 192 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 240 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 255 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 256 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 257 | 457 / 2048 | 9.537e-07 | ❌ |
| world=4，NCCL_ALGO=default，H=2048 | sparse_pos | float32 | 1 → 300 | 457 / 2048 | 9.537e-07 | ❌ |

### 表 T1b：批大小每加 1，未变的那些行是否还是原来的位

比较的是 `[b,H]` 与 `[b+1,H]` 的前 b 行（H=2048，与 30B-A3B 的 hidden 一致），
b 从 1 走到 299。下表统计有多少个 b→b+1 出现了位差，并列出最早几个。

| 运行 | 数据类型 | 出现位差的步数 | 最早的几个 b→b+1 |
|---|---|---|---|
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | bfloat16 | 0 / 299 | 无 |
| world=2，NCCL_ALGO=allreduce:ring，H=2048 | float32 | 0 / 299 | 无 |
| world=4，NCCL_ALGO=default，H=2048 | bfloat16 | 299 / 299 | 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 |
| world=4，NCCL_ALGO=default，H=2048 | float32 | 299 / 299 | 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | bfloat16 | 46 / 299 | 4→5, 6→7, 8→9, 10→11, 11→12, 15→16, 16→17, 20→21 |
| world=4，NCCL_ALGO=allreduce:tree，H=2048 | float32 | 81 / 299 | 2→3, 3→4, 4→5, 5→6, 7→8, 8→9, 10→11, 11→12 |

### 表 T2：vLLM eager TP=4 固定 NCCL 算法后的 32 条 prompt

| NCCL_ALGO | 位级一致 | token id 变化 | 最大 \|Δlogprob\| | 重复批不一致 | 批量耗时 |
|---|---|---|---|---|---|
| default | 1/32 | 10 | 2.723e+00 | 1 | 1.43 s |
| ring | 2/32 | 10 | 2.723e+00 | 1 | 1.18 s |
| tree | 2/32 | 10 | 2.723e+00 | 0 | 1.30 s |

### 表 T3：前缀缓存命中与 fresh prefill 随 tail 长度的变化

基座 prompt 6000 token，比较前 8 个生成 token。

| tail token | 位级一致 | 首个不一致 | 最大 \|Δlogprob\| | token id 相同 |
|---|---|---|---|---|
| 1 | ✅ | — | 0.0 | ✅ |
| 2 | ✅ | — | 0.0 | ✅ |
| 4 | ✅ | — | 0.0 | ✅ |
| 8 | ✅ | — | 0.0 | ✅ |
| 16 | ✅ | — | 0.0 | ✅ |
| 32 | ✅ | — | 0.0 | ✅ |
| 64 | ✅ | — | 0.0 | ✅ |
| 128 | ✅ | — | 0.0 | ✅ |
| 192 | ✅ | — | 0.0 | ✅ |
| 256 | ❌ | 0 | 3.576e-07 | ✅ |
| 320 | ✅ | — | 0.0 | ✅ |
| 384 | ✅ | — | 0.0 | ✅ |
| 448 | ❌ | 2 | 2.265e-06 | ✅ |
| 512 | ✅ | — | 0.0 | ✅ |

### 表 T4（compiled）：CUDA graph bucket 边界附近的批大小扫描

| 批大小 | 位级一致 | 首个不一致 | 最大 \|Δlogprob\| | 该批耗时 |
|---|---|---|---|---|
| 31 | ✅ | — | 0.0 | 1.10 s |
| 32 | ✅ | — | 0.0 | 1.07 s |
| 33 | ✅ | — | 0.0 | 1.22 s |
| 63 | ✅ | — | 0.0 | 1.32 s |
| 64 | ✅ | — | 0.0 | 1.48 s |
| 65 | ✅ | — | 0.0 | 1.60 s |
| 127 | ✅ | — | 0.0 | 1.91 s |
| 128 | ✅ | — | 0.0 | 1.79 s |
| 129 | ✅ | — | 0.0 | 1.86 s |
| 255 | ✅ | — | 0.0 | 2.30 s |
| 256 | ✅ | — | 0.0 | 2.17 s |
| 257 | ✅ | — | 0.0 | 3.10 s |

### 表 T4（eager）：CUDA graph bucket 边界附近的批大小扫描

| 批大小 | 位级一致 | 首个不一致 | 最大 \|Δlogprob\| | 该批耗时 |
|---|---|---|---|---|
| 31 | ✅ | — | 0.0 | 2.28 s |
| 32 | ✅ | — | 0.0 | 2.28 s |
| 33 | ✅ | — | 0.0 | 2.19 s |
| 63 | ✅ | — | 0.0 | 2.16 s |
| 64 | ✅ | — | 0.0 | 2.17 s |
| 65 | ✅ | — | 0.0 | 2.37 s |
| 127 | ✅ | — | 0.0 | 2.37 s |
| 128 | ✅ | — | 0.0 | 2.28 s |
| 129 | ✅ | — | 0.0 | 2.25 s |
| 255 | ✅ | — | 0.0 | 2.51 s |
| 256 | ❌ | 5 | 8.805e-02 | 2.74 s |
| 257 | ❌ | 5 | 8.805e-02 | 2.69 s |


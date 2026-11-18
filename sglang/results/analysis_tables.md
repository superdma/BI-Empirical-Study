## batchall

| 配置 | solo vs batch | batch vs batch | n |
|---|---|---|---|
| sgl_det | 25/32 不一致 | 0/32 不一致 | 32 |
| sgl_det_dcar | 25/32 不一致 | 0/32 不一致 | 32 |
| sgl_det_nograph | 25/32 不一致 | 0/32 不一致 | 32 |
| sgl_det_pp4 | 0/32 不一致 | 0/32 不一致 | 32 |
| sgl_det_pp4_graph | 0/32 不一致 | 0/32 不一致 | 32 |
| sgl_det_triton | 17/32 不一致 | 0/32 不一致 | 32 |
| sgl_nodet | 32/32 不一致 | 0/32 不一致 | 32 |
| smoke_det | 0/32 不一致 | 0/32 不一致 | 32 |
| smoke_det_pp4 | 0/32 不一致 | 0/32 不一致 | 32 |
| smoke_det_tp2 | 0/32 不一致 | 0/32 不一致 | 32 |
| smoke_det_tp4 | 32/32 不一致 | 0/32 不一致 | 32 |
| smoke_det_tp4_triton | 32/32 不一致 | 0/32 不一致 | 32 |

## bssweep

| bs | sgl_det | sgl_det_dcar | sgl_det_nograph | sgl_det_pp4 | sgl_det_pp4_graph | sgl_det_triton | sgl_nodet | smoke_det | smoke_det_pp4 | smoke_det_tp2 | smoke_det_tp4 | smoke_det_tp4_triton |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 6.740e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 192 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 2.187e-01 ❌ (ids ✗) | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 224 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 2.187e-01 ❌ (ids ✗) | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 240 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 2.187e-01 ❌ (ids ✗) | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 255 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 2.187e-01 ❌ (ids ✗) | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 256 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 2.187e-01 ❌ (ids ✗) | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |
| 257 | 4.083e-02 ❌ | 4.083e-02 ❌ | 4.083e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 5.003e-02 ❌ | 6.324e-02 ❌ | 0.0 ✅ | 0.0 ✅ | 0.0 ✅ | 1.347e+00 ❌ (ids ✗) | 6.868e-01 ❌ (ids ✗) |

## len

| 配置 | solo vs batch | batch vs batch | 总 token |
|---|---|---|---|
| sgl_det_dcar | 32/32 不一致 | 0/32 不一致 | 284 |
| sgl_det | 32/32 不一致 | 0/32 不一致 | 284 |
| sgl_det_nograph | 32/32 不一致 | 0/32 不一致 | 284 |
| sgl_det_pp4_graph | 0/32 不一致 | 0/32 不一致 | 284 |
| sgl_det_pp4 | 0/32 不一致 | 0/32 不一致 | 284 |
| sgl_det_triton | 31/32 不一致 | 0/32 不一致 | 284 |
| sgl_nodet | 32/32 不一致 | 0/32 不一致 | 284 |
| smoke_det | 0/32 不一致 | 0/32 不一致 | 284 |
| smoke_det_pp4 | 0/32 不一致 | 0/32 不一致 | 284 |
| smoke_det_tp2 | 0/32 不一致 | 0/32 不一致 | 284 |
| smoke_det_tp4 | 32/32 不一致 | 0/32 不一致 | 284 |
| smoke_det_tp4_triton | 31/32 不一致 | 0/32 不一致 | 284 |

## refeed

| 配置 | 比较数 | 不一致 | 首个不一致位置 | 最大 |Δlogprob| |
|---|---|---|---|---|
| sgl_det_dcar | 64 | 64 | 0 | 8.106e-02 |
| sgl_det_nograph | 64 | 64 | 0 | 8.106e-02 |
| sgl_det_pp4_graph | 64 | 51 | 12 | 1.141e-01 |
| sgl_det_pp4 | 64 | 51 | 12 | 1.141e-01 |
| sgl_det | 64 | 64 | 0 | 8.106e-02 |
| sgl_det_triton | 64 | 64 | 0 | 2.049e-01 |
| sgl_nodet | 64 | 62 | 0 | 1.577e-01 |
| smoke_det_pp4 | 64 | 64 | 0 | 9.853e-02 |
| smoke_det | 64 | 64 | 0 | 9.853e-02 |
| smoke_det_tp2 | 64 | 64 | 0 | 8.229e-02 |
| smoke_det_tp4 | 64 | 64 | 0 | 6.990e-02 |
| smoke_det_tp4_triton | 64 | 64 | 0 | 5.826e-02 |

## long

| 配置 | fresh vs cache hit | hit vs hit2 |
|---|---|---|
| sgl_det_dcar | 0.0 ✅ | 0.0 ✅ |
| sgl_det | 0.0 ✅ | 0.0 ✅ |
| sgl_det_nograph | 0.0 ✅ | 0.0 ✅ |
| sgl_det_pp4_graph | 0.0 ✅ | 0.0 ✅ |
| sgl_det_pp4 | 0.0 ✅ | 0.0 ✅ |
| sgl_det_triton | 1.192e-07 ❌ | 0.0 ✅ |
| sgl_nodet | 1.192e-07 ❌ | 0.0 ✅ |
| smoke_det | 0.0 ✅ | 0.0 ✅ |
| smoke_det_pp4 | 0.0 ✅ | 0.0 ✅ |
| smoke_det_tp2 | 0.0 ✅ | 0.0 ✅ |
| smoke_det_tp4 | 0.0 ✅ | 0.0 ✅ |
| smoke_det_tp4_triton | 5.076e-05 ❌ | 0.0 ✅ |

## http

| 场景 | 比较数 | 不一致 |
|---|---|---|
| seq vs concurrent | 96 | 60 |
| identical vs baseline | 96 | 64 |
| seq vs burst | 96 | 57 |

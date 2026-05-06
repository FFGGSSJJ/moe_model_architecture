# `nvbandwidth` results (formatted for markdown)

**NOTE: this results tell us that it seems we have a lower D2H speed compare to what is said here:  https://dnhkng.github.io/posts/gh200-benchmarking/ and https://www.youtube.com/watch?v=dPULokuSuPQ**

## System Info

| Field | Value |
|---|---|
| `nvbandwidth` Version | `v0.9` |
| Git Version | `v0.9` |
| CUDA Runtime Version | `13010` |
| CUDA Driver Version | `13010` |
| Driver Version | `590.48.01` |
| Host | `nid006272` |

## Devices

| Device | GPU | PCI Bus ID |
|---:|---|---|
| 0 | NVIDIA GH200 120GB | `00000009:01:00` |
| 1 | NVIDIA GH200 120GB | `00000019:01:00` |
| 2 | NVIDIA GH200 120GB | `00000029:01:00` |
| 3 | NVIDIA GH200 120GB | `00000039:01:00` |

---

## CE memcpy bandwidth

### `host_to_device_memcpy_ce`

CPU(row) → GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 422.07 | 420.60 | 422.13 | 420.99 |

**SUM:** `1685.79`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_host_memcpy_ce`

CPU(row) ← GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 170.29 | 170.06 | 170.23 | 170.05 |

**SUM:** `680.63`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `host_to_device_bidirectional_memcpy_ce`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 340.87 | 340.03 | 340.59 | 339.94 |

**SUM:** `1361.43`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_host_bidirectional_memcpy_ce`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 149.71 | 149.36 | 149.62 | 149.30 |

**SUM:** `597.99`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Device-to-device CE memcpy bandwidth

### `device_to_device_memcpy_read_ce`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 133.02 | 132.98 | 132.98 |
| 1 | 132.98 | N/A | 132.94 | 132.94 |
| 2 | 132.94 | 132.98 | N/A | 127.36 |
| 3 | 127.03 | 127.19 | 127.10 | N/A |

**SUM:** `1572.45`  
**COEFFICIENT_OF_VARIATION:** `0.02`

### `device_to_device_memcpy_write_ce`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 133.25 | 133.25 | 133.22 |
| 1 | 133.26 | N/A | 133.25 | 133.22 |
| 2 | 133.26 | 133.25 | N/A | 133.22 |
| 3 | 133.26 | 133.25 | 133.22 | N/A |

**SUM:** `1598.91`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_read_ce_read1`

GPU(row) ↔ GPU(column), Read1 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 127.12 | 127.05 | 127.33 |
| 1 | 127.05 | N/A | 132.17 | 132.19 |
| 2 | 132.17 | 132.20 | N/A | 127.35 |
| 3 | 127.05 | 127.19 | 127.10 | N/A |

**SUM:** `1545.97`  
**COEFFICIENT_OF_VARIATION:** `0.02`

### `device_to_device_bidirectional_memcpy_read_ce_read2`

GPU(row) ↔ GPU(column), Read2 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 127.12 | 132.17 | 127.07 |
| 1 | 127.21 | N/A | 132.23 | 127.42 |
| 2 | 126.60 | 132.18 | N/A | 127.23 |
| 3 | 126.78 | 132.19 | 126.80 | N/A |

**SUM:** `1545.00`  
**COEFFICIENT_OF_VARIATION:** `0.02`

### `device_to_device_bidirectional_memcpy_read_ce_total`

GPU(row) ↔ GPU(column), Total bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 254.24 | 259.22 | 254.40 |
| 1 | 254.26 | N/A | 264.40 | 259.61 |
| 2 | 258.77 | 264.38 | N/A | 254.58 |
| 3 | 253.83 | 259.38 | 253.90 | N/A |

**SUM:** `3090.97`  
**COEFFICIENT_OF_VARIATION:** `0.01`

### `device_to_device_bidirectional_memcpy_write_ce_write1`

GPU(row) ↔ GPU(column), Write1 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 132.59 | 132.58 | 132.59 |
| 1 | 132.59 | N/A | 132.49 | 132.48 |
| 2 | 132.48 | 132.47 | N/A | 132.55 |
| 3 | 132.55 | 132.55 | 132.55 | N/A |

**SUM:** `1590.44`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_write_ce_write2`

GPU(row) ↔ GPU(column), Write2 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 132.58 | 132.48 | 132.55 |
| 1 | 132.59 | N/A | 132.47 | 132.53 |
| 2 | 132.58 | 132.47 | N/A | 132.55 |
| 3 | 132.56 | 132.48 | 132.54 | N/A |

**SUM:** `1590.39`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_write_ce_total`

GPU(row) ↔ GPU(column), Total bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 265.17 | 265.06 | 265.13 |
| 1 | 265.17 | N/A | 264.96 | 265.02 |
| 2 | 265.06 | 264.94 | N/A | 265.09 |
| 3 | 265.11 | 265.02 | 265.09 | N/A |

**SUM:** `3180.83`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Aggregate CE memcpy bandwidth

### `all_to_host_memcpy_ce`

CPU(row) ← GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 170.25 | 170.00 | 170.28 | 169.99 |

**SUM:** `680.52`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `all_to_host_bidirectional_memcpy_ce`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 149.70 | 149.34 | 149.58 | 149.35 |

**SUM:** `597.97`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `host_to_all_memcpy_ce`

CPU(row) → GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 421.49 | 420.50 | 422.09 | 421.01 |

**SUM:** `1685.09`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `host_to_all_bidirectional_memcpy_ce`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 340.79 | 340.03 | 340.64 | 339.83 |

**SUM:** `1361.30`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `all_to_one_write_ce`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 399.66 | 399.64 | 399.56 | 399.54 |

**SUM:** `1598.39`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `all_to_one_read_ce`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 113.89 | 113.93 | 113.88 | 113.88 |

**SUM:** `455.58`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `one_to_all_write_ce`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 398.73 | 398.78 | 398.78 | 398.78 |

**SUM:** `1595.07`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `one_to_all_read_ce`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 380.95 | 381.45 | 376.90 | 381.23 |

**SUM:** `1520.54`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## SM memcpy bandwidth

### `host_to_device_memcpy_sm`

CPU(row) → GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 410.01 | 401.34 | 414.43 | 409.64 |

**SUM:** `1635.42`  
**COEFFICIENT_OF_VARIATION:** `0.01`

### `device_to_host_memcpy_sm`

CPU(row) ← GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 380.12 | 379.93 | 380.46 | 380.22 |

**SUM:** `1520.73`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `host_to_device_bidirectional_memcpy_sm`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 290.34 | 251.76 | 300.98 | 289.38 |

**SUM:** `1132.45`  
**COEFFICIENT_OF_VARIATION:** `0.07`

### `device_to_host_bidirectional_memcpy_sm`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 289.01 | 251.23 | 295.61 | 289.36 |

**SUM:** `1125.21`  
**COEFFICIENT_OF_VARIATION:** `0.06`

---

## Device-to-device SM memcpy bandwidth

### `device_to_device_memcpy_read_sm`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 125.71 | 125.70 | 125.70 |
| 1 | 125.74 | N/A | 125.73 | 125.69 |
| 2 | 125.73 | 125.73 | N/A | 125.75 |
| 3 | 125.74 | 125.73 | 125.74 | N/A |

**SUM:** `1508.69`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_memcpy_write_sm`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 124.60 | 124.59 | 124.59 |
| 1 | 124.61 | N/A | 124.62 | 124.61 |
| 2 | 124.58 | 124.60 | N/A | 124.60 |
| 3 | 124.59 | 124.61 | 124.59 | N/A |

**SUM:** `1495.19`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_read_sm_read1`

GPU(row) ↔ GPU(column), Read1 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 113.18 | 113.19 | 113.18 |
| 1 | 113.16 | N/A | 113.21 | 113.18 |
| 2 | 113.19 | 113.12 | N/A | 113.20 |
| 3 | 113.20 | 113.18 | 113.15 | N/A |

**SUM:** `1358.15`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_read_sm_read2`

GPU(row) ↔ GPU(column), Read2 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 113.21 | 113.20 | 113.20 |
| 1 | 113.18 | N/A | 113.20 | 113.19 |
| 2 | 113.17 | 113.15 | N/A | 113.19 |
| 3 | 113.23 | 113.26 | 113.25 | N/A |

**SUM:** `1358.43`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_read_sm_total`

GPU(row) ↔ GPU(column), Total bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 226.40 | 226.38 | 226.38 |
| 1 | 226.34 | N/A | 226.41 | 226.37 |
| 2 | 226.36 | 226.28 | N/A | 226.39 |
| 3 | 226.42 | 226.44 | 226.41 | N/A |

**SUM:** `2716.58`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_write_sm_write1`

GPU(row) ↔ GPU(column), Write1 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 123.90 | 123.88 | 123.85 |
| 1 | 123.86 | N/A | 123.89 | 123.91 |
| 2 | 123.91 | 123.88 | N/A | 123.87 |
| 3 | 123.89 | 123.92 | 123.92 | N/A |

**SUM:** `1486.69`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_write_sm_write2`

GPU(row) ↔ GPU(column), Write2 bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 123.87 | 123.91 | 123.89 |
| 1 | 123.93 | N/A | 123.92 | 123.91 |
| 2 | 123.89 | 123.91 | N/A | 123.90 |
| 3 | 123.92 | 123.90 | 123.94 | N/A |

**SUM:** `1486.89`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_bidirectional_memcpy_write_sm_total`

GPU(row) ↔ GPU(column), Total bandwidth, GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 247.77 | 247.79 | 247.73 |
| 1 | 247.79 | N/A | 247.81 | 247.82 |
| 2 | 247.80 | 247.79 | N/A | 247.77 |
| 3 | 247.81 | 247.82 | 247.86 | N/A |

**SUM:** `2973.58`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Aggregate SM memcpy bandwidth

### `all_to_host_memcpy_sm`

CPU(row) ← GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 380.24 | 379.88 | 380.24 | 380.19 |

**SUM:** `1520.54`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `all_to_host_bidirectional_memcpy_sm`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 292.17 | 251.52 | 299.70 | 291.50 |

**SUM:** `1134.89`  
**COEFFICIENT_OF_VARIATION:** `0.07`

### `host_to_all_memcpy_sm`

CPU(row) → GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 410.48 | 401.72 | 414.01 | 411.43 |

**SUM:** `1637.64`  
**COEFFICIENT_OF_VARIATION:** `0.01`

### `host_to_all_bidirectional_memcpy_sm`

CPU(row) ↔ GPU(column), GB/s

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 292.21 | 252.10 | 299.91 | 290.89 |

**SUM:** `1135.11`  
**COEFFICIENT_OF_VARIATION:** `0.07`

### `all_to_one_write_sm`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 373.65 | 373.68 | 373.58 | 373.58 |

**SUM:** `1494.50`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `all_to_one_read_sm`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 136.62 | 139.79 | 139.82 | 140.09 |

**SUM:** `556.33`  
**COEFFICIENT_OF_VARIATION:** `0.01`

### `one_to_all_write_sm`

GPU(row) → GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 136.03 | 139.95 | 138.46 | 138.62 |

**SUM:** `553.06`  
**COEFFICIENT_OF_VARIATION:** `0.01`

### `one_to_all_read_sm`

GPU(row) ← GPU(column), GB/s

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 373.62 | 373.65 | 373.65 | 373.60 |

**SUM:** `1494.53`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Latency

### `host_device_latency_sm`

CPU(row) ↔ GPU(column), ns

| CPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 618.71 | 617.16 | 617.44 | 618.43 |

**SUM:** `2471.74`  
**COEFFICIENT_OF_VARIATION:** `0.00`

### `device_to_device_latency_sm`

GPU(row) ↔ GPU(column), ns

| GPU \ GPU | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | N/A | 545.18 | 543.93 | 543.13 |
| 1 | 546.20 | N/A | 546.84 | 547.65 |
| 2 | 547.55 | 547.10 | N/A | 542.49 |
| 3 | 545.21 | 546.27 | 543.13 | N/A |

**SUM:** `6544.67`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Local device copy

### `device_local_copy`

GPU(column), GB/s

| Row | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 1799.32 | 1800.07 | 1800.07 | 1801.20 |

**SUM:** `7200.66`  
**COEFFICIENT_OF_VARIATION:** `0.00`

---

## Note

The reported results may not reflect the full capabilities of the platform.

Performance can vary with software drivers, hardware clocks, and system topology.
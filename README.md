# GPU Broad Phase Collision Detection

CUDA Uniform Grid broad phase collision detection algorithms from the paper *"Comparative Performance Evaluation of CPU Sweep and Prune and GPU Uniform Grid Broad Phase Algorithms"*.

* **Strategy A (Multi-Cell):** Inserts objects into all overlapping cells. Robust for high geometric variance.
* **Strategy B (Single-Cell):** Assigns objects to one cell. Uses our optimized **14-cell Half Shell** or traditional 27-cell neighbor search. Best for uniform particle systems.

## Integration

Builds as a static library (`broad_phase_cuda`) via CMake (3.18+). Requires CUDA Toolkit (Standard 17) and CUB. Grid parameters are exposed as CMake cache variables:

```cmake
set(GRID_CELL_SIZE "1.0f")
set(GRID_RES_X "100")
set(GRID_RES_Y "100")
set(GRID_RES_Z "100")
```

## Reproducing Paper Benchmarks

The benchmark suite is hosted in the [Tics physics engine](https://github.com/timo-eberl/tics) repository. To reproduce the setup used for the paper, use the commit [`710b18d`](https://github.com/timo-eberl/tics/tree/710b18db149dabbc73180a2c379cb62c6c6fd631).

```bash
# Setup Tics repository
git clone https://github.com/timo-eberl/tics.git
cd tics
git checkout 710b18db149dabbc73180a2c379cb62c6c6fd631

# Build benchmarks
./demos/benchmark_broadphase/build_benchmark_1.sh
./demos/benchmark_broadphase/build_benchmark_2.sh

# Run benchmarks (outputs raw logs to results/)
./demos/benchmark_broadphase/run_benchmark_1.sh
./demos/benchmark_broadphase/run_benchmark_2.sh

# Parse results into PGFPlots-compatible coordinate tuples
python3 demos/benchmark_broadphase/parse_benchmark_1.py
python3 demos/benchmark_broadphase/parse_benchmark_2.py
```

The final parsed data will be written to `parsed_benchmark1.txt` and `parsed_benchmark2.txt`.

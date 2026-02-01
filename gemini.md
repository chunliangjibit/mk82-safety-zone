# Gemini Project Overview: Mk82 Safe Zone v2.0

## System Personality
This system is designed as a **High-Performance Safety Calculation Suite**. It prioritizes computational speed and professional visualization, bridging the gap between raw fragmentation data and tactical decision-making support.

## Architecture Meta-Data
- **Environment**: Conda `swarm` (Windows).
- **Core Technology**: Python 3.x + Numba (JIT) + NumPy + Multiprocessing.
- **Design Pattern**: Data Ingestion -> Binary Caching -> JIT Kernel -> Multi-process Scanning -> Multi-view Reporting.

## Key Logic Blocks for Gemini
1.  **`core_solver.py`**: The "Physics Brain". Uses Numba to compile Python to machine code. Uses bisection (Binary Search) to solve for safety distance, ensuring $O(\log n)$ efficiency.
2.  **`batch_runner.py`**: The "Conductor". Spawns worker processes to saturate CPU cores. Aggregates worst-case envelopes into a global maximal surface.
3.  **`viz_envelope.py`**: The "Artist". Generates standardized engineering PNGs and interactive HTML.
4.  **`config.yaml`**: The "Single Source of Truth". All constants and scan ranges live here.

## Data Schema
- `.npy` files in `data_cache` store fragment info as $[Mass, Count, V_{static}, Angle_{static}]$.
- Results are stored as $[n_{\theta}, n_{\phi}]$ grids representing the radial distance $R$.

## Future Extensibility
- **New Platforms**: To support new aircraft, adjust the `target:area` and `target:safe_prob` in `config.yaml`.
- **New Munitions**: Re-run `ingest_data.py` on any Markdown formatted table following the C32/C33 standard.

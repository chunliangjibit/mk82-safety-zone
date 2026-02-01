# Mk82/BLU-111 Safety Envelope Calculation System

A high-performance Python suite for calculating 1e-4 probability fragmentation safety envelopes based on US Manual Tables C-32 and C-33.

## Features
- **HPC Physics**: Numba JIT accelerated aerodynamics and probability flux solver.
- **Parallel Processing**: Multiprocessing grid search for rapid envelope generation across multiple flight profiles.
- **Professional Viz**: Standardized engineering 4-view plots, interactive 3D HTML models, and OBJ mesh export.
- **Dual Mode Support**: Seamlessly switch between Nose Initiation (Mk 82) and Tail Initiation (BLU-111/B).

## Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure**:
   Edit `config.yaml` to set initiation mode (`Nose` or `Tail`).
3. **Execute**:
   ```bash
   python batch_runner.py
   python viz_envelope.py
   ```

## Output
Results are stored in `data_cache/`:
- `envelope_report_*.txt`: Detailed tactical summary.
- `envelope_global.png`: Engineering 4-view plot.
- `envelope_3d.html`: Interactive 3D visualization.

## Final Results (v4.0 Production)
For **BLU-111 (Tail Initiation)** at sea level:
- **Consolidated Safe Distance**: **880.75 m**
- (Worst-case safety boundary across all aspects)

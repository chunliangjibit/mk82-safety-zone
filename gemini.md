# Gemini Project Overview: Safe Zone v5.1 (Tactical)

## System Personality
This system is a **High-Performance Tactical Safety Suite**. It uses physics-based dynamic projection (Box Model) to move beyond conservative static assumptions, providing both "Absolute Redlines" and "Tactical Recommendations".

## Architecture Meta-Data
- **Environment**: Conda `swarm` (Windows).
- **Core Technology**: Python 3.x + Numba (JIT) + NumPy + Multiprocessing.
- **Final Logic**: Supports **Nose/Tail** initiation modes with automatic dataset selection. Outputs a single **Omnidirectional Global Max** safety distance.

## Evolution History
1.  **v1.0**: Research & Data Parsing.
2.  **v2.0**: HPC Engine (Numba) implementation.
3.  **v3.0**: Tail-aspect corridor research (Audit fix).
4.  **v4.0**: Consolidation to absolute omnidirectional safety (Final Delivery).
5.  **v5.0**: Tactical Core Update. Multi-aspect projection (Box Model) & Height-filtered analysis.

## Key Logic Blocks
1.  **`core_solver.py`**: JIT Physics Engine.
2.  **`batch_runner.py`**: Parallel Grid Search.
3.  **`viz_envelope.py`**: Standardized Global Visualization.
4.  **`generate_report.py`**: Consolidated Tactical Reporting.

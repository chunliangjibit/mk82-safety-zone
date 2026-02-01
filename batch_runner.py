
import numpy as np
import yaml
import itertools
import multiprocessing
import os
import time
from tqdm import tqdm
from functools import partial

# Import solver (ensure it's importable)
from core_solver import solve_single_case

def worker(task, config, fragments, physics_cache=None):
    """
    Worker function for multiprocessing.
    task: (v_bomb, angle_bomb)
    """
    v, angle = task
    # Solve single case
    # This returns a 2D grid (Theta x Phi) of Safe Distances
    envelope = solve_single_case(v, angle, fragments, config)
    return envelope

def main():
    # Load Config
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Load Data
    # We process Mk82 first (or make it selectable)
    # Using Mk82 for now
    frag_path = os.path.join(config['data']['output_dir'], config['data']['mk82_file'])
    print(f"Loading Fragments from {frag_path}...")
    fragments = np.load(frag_path)
    
    # Generate Tasks
    scan = config['scan']
    v_min, v_max, v_step = scan['velocity']['min'], scan['velocity']['max'], scan['velocity']['step']
    a_min, a_max, a_step = scan['angle']['min'], scan['angle']['max'], scan['angle']['step']
    
    # inclusive ranges
    velocities = np.arange(v_min, v_max + 0.01, v_step)
    angles = np.arange(a_min, a_max + 0.01, a_step)
    
    tasks = list(itertools.product(velocities, angles))
    print(f"Total Tasks: {len(tasks)}")
    
    # Prepare Output Aggregation
    c_cmp = config['compute']
    n_theta = int(c_cmp['spatial_bins'])
    n_phi = int(c_cmp['spatial_bins'])
    
    # Global Max Envelope
    # We want Max(SafeDist) per (Theta, Phi) over ALL tasks.
    global_max_envelope = np.zeros((n_theta, n_phi), dtype=np.float64)
    
    # Parallel Execution
    n_workers = int(c_cmp['parallel_workers'])
    if n_workers < 0:
        n_workers = max(1, os.cpu_count() + n_workers)
        
    print(f"Starting Pool with {n_workers} workers...")
    
    # We need to workaround pickle issues with Numba?
    # Usually compiled functions pickling works if imported.
    # But arguments must be picklable. Numpy arrays are fine. Config dict is fine.
    
    # Create partial to freeze constant args
    # Note: fragments is large-ish (few KB/MB), passing it to every worker is fine via Fork (Linux) 
    # but on Windows 'spawn', it gets pickled. 
    # 200 rows is tiny.
    
    func = partial(worker, config=config, fragments=fragments)
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Use imap to get progress bar
        results = list(tqdm(pool.imap(func, tasks), total=len(tasks), unit="case"))
        
    print(f"Compute Finished in {time.time() - start_time:.2f}s. Aggregating...")
    
    # Aggregate
    for res in results:
        global_max_envelope = np.maximum(global_max_envelope, res)
        
    # Save Result
    out_file = os.path.join(config['data']['output_dir'], "envelope_result.npy")
    np.save(out_file, global_max_envelope)
    print(f"Saved Global Envelope to {out_file}")
    print(f"Max Overall Distance: {np.max(global_max_envelope):.2f} m")
    
    # Generate Report
    from generate_report import generate_calculation_report
    generate_calculation_report("config.yaml", out_file, config['data']['output_dir'])

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()

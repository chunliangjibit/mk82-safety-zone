
import numpy as np
import yaml
import time
from core_solver import solve_single_case

def main():
    print("Loading Config...")
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print("Loading Data...")
    frag_path = "data_cache/mk82_fragments.npy"
    fragments = np.load(frag_path)
    print(f"Fragments: {fragments.shape}")
    
    # Test Parameters
    v_bomb = 260.0 # m/s
    angle_bomb = 85.0 # deg (Dive)
    
    print(f"Running Single Case: V={v_bomb} m/s, Angle={angle_bomb} deg")
    
    start_t = time.time()
    
    # Run twice to gauge JIT time
    envelope = solve_single_case(v_bomb, angle_bomb, fragments, config)
    
    jit_time = time.time() - start_t
    print(f"First Run (JIT overhead): {jit_time:.4f} s")
    
    start_t = time.time()
    envelope = solve_single_case(v_bomb, angle_bomb, fragments, config)
    run_time = time.time() - start_t
    print(f"Second Run (Compiled): {run_time:.4f} s")
    
    print("--- Results ---")
    print(f"Grid Shape: {envelope.shape}")
    print(f"Max Safe Distance: {np.max(envelope):.2f} m")
    print(f"Mean Safe Distance: {np.mean(envelope):.2f} m")
    
    # Check "Forward" vs "Rear"
    # Forward is usually Theta ~ 90 (if horizontal) or aligned with Bomb Vector.
    # In our coords, Bomb Vector is determined by Angle.
    # If Angle=85, Bomb is pointing Down (-Z).
    # So "Forward" is Pole Pi (Theta=Pi).
    # "Rear" is Theta=0.
    
    # Pole Indices
    n_theta = envelope.shape[0]
    
    r_rear = np.mean(envelope[0:5, :]) # Top of sphere (Up)
    r_nose = np.mean(envelope[-5:, :]) # Bottom of sphere (Down)
    
    print(f"Safe Distance Near Rear (Up): {r_rear:.2f} m")
    print(f"Safe Distance Near Nose (Down): {r_nose:.2f} m")
    
if __name__ == "__main__":
    main()


import numpy as np
import yaml
import os

GRAINS_TO_KG = 6.479891e-5

def check_file(path, name):
    if not os.path.exists(path):
        print(f"File {path} invalid.")
        return
        
    data = np.load(path)
    # Data columns: [mass_kg, count, velocity_ms, angle_rad]
    
    total_count = np.sum(data[:, 1])
    total_mass_kg = np.sum(data[:, 0] * data[:, 1]) # Total mass of all fragments? 
    # Wait, the data rows are "Mean Mass" and "Count".
    # So total mass = sum(MeanMass * Count).
    
    total_mass_grains = total_mass_kg / GRAINS_TO_KG
    
    print(f"--- {name} ---")
    print(f"Rows: {len(data)}")
    print(f"Total Count: {total_count:.1f}")
    print(f"Total Mass: {total_mass_grains:,.1f} grains ({total_mass_kg:.2f} kg)")
    print(f"Velocity Range: {np.min(data[:, 2]):.1f} - {np.max(data[:, 2]):.1f} m/s")
    
def main():
    print("Verifying Data...")
    check_file("data_cache/mk82_fragments.npy", "Mk82 Tritonal")
    check_file("data_cache/blu111_fragments.npy", "BLU-111 PBXN-109")

if __name__ == "__main__":
    main()

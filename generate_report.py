
import yaml
import os
import datetime
import numpy as np

def generate_calculation_report(config_path, data_path, output_dir):
    """
    Generates a detailed text report of the calculation.
    """
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Load Results
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return
        
    envelope = np.load(data_path)
    max_safe_dist = np.max(envelope)
    
    # Generate Filename with Timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"envelope_report_{ts}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========================================================\n")
        f.write("       MK82 SAFETY ENVELOPE CALCULATION REPORT          \n")
        f.write("========================================================\n\n")
        
        f.write(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config File: {config_path}\n")
        f.write(f"Result Data: {data_path}\n\n")
        
        f.write("--------------------------------------------------------\n")
        f.write("1. INPUT PARAMETERS\n")
        f.write("--------------------------------------------------------\n")
        
        c = config['bomb']
        f.write("[Bomb Physics]\n")
        f.write(f"  Mass:         {c['mass']} kg\n")
        f.write(f"  Shape Factor: {c['shape_factor']}\n")
        f.write(f"  Drag Coeff:   {c['drag_coeff']}\n")
        f.write(f"  Air Density:  {c['air_density']} kg/m^3\n\n")
        
        t = config['target']
        f.write("[Safety Criteria]\n")
        f.write(f"  Vulnerable Area:  {t['area']} m^2\n")
        f.write(f"  Safety Prob:      {t['safe_prob']}\n")
        f.write(f"  Lethal Energy:    {t['energy_threshold']} J\n\n")
        
        s = config['scan']
        f.write("[Scan Envelope]\n")
        f.write(f"  Velocity: {s['velocity']['min']} - {s['velocity']['max']} m/s (Step: {s['velocity']['step']})\n")
        f.write(f"  Angle:    {s['angle']['min']} - {s['angle']['max']} deg (Step: {s['angle']['step']})\n\n")
        
        f.write("--------------------------------------------------------\n")
        f.write("2. DATA SOURCE\n")
        f.write("--------------------------------------------------------\n")
        frag_file = os.path.join(config['data']['output_dir'], config['data']['mk82_file'])
        if os.path.exists(frag_file):
            frags = np.load(frag_file)
            f.write(f"Fragment Database: {config['data']['mk82_file']}\n")
            f.write(f"  Total Fragment Groups: {len(frags)}\n")
            f.write(f"  Mean Fragment Count:   {np.sum(frags[:,1]):.1f}\n")
        else:
            f.write("Fragment Data: Not Found\n")
        f.write("\n")
            
        f.write("--------------------------------------------------------\n")
        f.write("3. CALCULATION RESULTS\n")
        f.write("--------------------------------------------------------\n")
        
        # 1. Global Max Hazard
        max_safe_dist = np.max(envelope)
        f.write(f"GLOBAL MAX HAZARD DISTANCE: {max_safe_dist:.4f} meters\n")
        f.write(f"  (Worst case: Side/Front aspect - Ground hazard)\n\n")
        
        # 2. Aircraft Safe Separation (Rear Aspect)
        # In Dive (75-90 deg), Theta 0 is Tailward (+Z).
        # We take the representative safe distance in the tail cone.
        n_theta, n_phi = envelope.shape
        rear_sector_deg = 30.0
        rear_idx_limit = int(n_theta * (rear_sector_deg / 180.0))
        
        # Use mean or a tighter percentile if max is noisy due to aggregation
        rear_sector_data = envelope[0:rear_idx_limit, :]
        dist_aircraft_safe = np.percentile(rear_sector_data, 95) # 95th percentile to avoid outliers
        
        f.write(f">>> AIRCRAFT SAFE SEPARATION: {dist_aircraft_safe:.4f} m <<<\n")
        f.write(f"  (Tail Aspect < {rear_sector_deg} deg - Representative safe distance)\n\n")

        f.write("[Grid Statistics]\n")
        f.write(f"  Resolution: {config['compute']['spatial_bins']} x {config['compute']['spatial_bins']}\n")
        f.write(f"  Mean Safe Distance: {np.mean(envelope):.4f} m\n")
        f.write(f"  Min Safe Distance:  {np.min(envelope):.4f} m\n\n")
        
        f.write("--------------------------------------------------------\n")
        f.write("END OF REPORT\n")
        f.write("========================================================\n")
        
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    # Test logic
    generate_calculation_report("config.yaml", "data_cache/envelope_result.npy", "data_cache")

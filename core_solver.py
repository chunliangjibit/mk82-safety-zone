
import numpy as np
import yaml
from numba import njit, prange
import math

# Load Config Defaults (Will be overwritten by config file if needed, but constants are baked for JIT)
# Ideally we pass constants to JIT functions
# Constants
GRAINS_TO_KG = 6.479891e-5
DEFAULT_RHO_FRAG = 7850.0 # kg/m^3 Steel

@njit(fastmath=True)
def get_decay_constant(mass_kg, shape_factor, drag_coeff, air_density, frag_density):
    # Area = SF * Vol^(2/3)
    # Vol = Mass / Rho
    vol = mass_kg / frag_density
    area = shape_factor * (vol ** (2.0/3.0))
    # k = (0.5 * rho_air * Cd * A) / m
    k = (0.5 * air_density * drag_coeff * area) / mass_kg
    return k

@njit(fastmath=True)
def calculate_d_stop(v0, k, mass, energy_threshold):
    # E = 0.5 * m * v^2
    # v_limit = sqrt(2 * E / m)
    # v(x) = v0 * exp(-k*x)
    # v_lim = v0 * exp(-k*D)
    # log(v_lim/v0) = -k*D
    # D = -1/k * log(v_lim/v0) = 1/k * log(v0/v_lim)
    
    val_limit_sq = 2 * energy_threshold / mass
    # if v0^2 < limit, distance is 0
    if (v0 * v0) <= val_limit_sq:
        return 0.0
    
    v_limit = math.sqrt(val_limit_sq)
    d = (1.0 / k) * math.log(v0 / v_limit)
    return d

@njit(fastmath=True)
def spherical_bin(px, py, pz):
    # Returns theta (0..pi), phi (-pi..pi)
    r = math.sqrt(px*px + py*py + pz*pz)
    if r == 0:
        return 0.0, 0.0
    theta = math.acos(pz / r) # 0 is +Z (Vertical up if Z is up)
    phi = math.atan2(py, px)
    return theta, phi

@njit(fastmath=True, parallel=False)
def generate_rays_jit(
    fragments, # [mass, count, v_static, angle_static]
    v_bomb_mag,
    angle_bomb_deg,
    shape_factor, drag_coeff, air_density, frag_density,
    energy_threshold,
    n_azimuth_sim
):
    # Returns Array [theta_g, phi_g, d_stop, weight]
    # Estimate size
    num_frags = len(fragments)
    total_rays = num_frags * n_azimuth_sim
    out_rays = np.zeros((total_rays, 4), dtype=np.float64)
    
    rad_bomb = math.radians(angle_bomb_deg)
    vb_x = v_bomb_mag * math.cos(rad_bomb)
    vb_z = -v_bomb_mag * math.sin(rad_bomb)
    vb_y = 0.0
    
    # Pre-calc bomb basis for rotation (W is Axis)
    v_mag = math.sqrt(vb_x*vb_x + vb_z*vb_z)
    if v_mag == 0:
        wx, wz = 1.0, 0.0
    else:
        wx = vb_x / v_mag
        wz = vb_z / v_mag
    
    count_idx = 0
    
    for i in range(num_frags):
        mass = fragments[i, 0]
        base_count = fragments[i, 1]
        v_stat = fragments[i, 2]
        ang_stat = fragments[i, 3]
        
        weight = base_count / n_azimuth_sim
        k = get_decay_constant(mass, shape_factor, drag_coeff, air_density, frag_density)
        
        vl_z = v_stat * math.cos(ang_stat)
        vl_rho = v_stat * math.sin(ang_stat)
        
        for j in range(n_azimuth_sim):
            phi_sim = (2.0 * math.pi * j) / n_azimuth_sim
            
            vl_x = vl_rho * math.cos(phi_sim)
            vl_y = vl_rho * math.sin(phi_sim)
            
            # Global Transform
            # W=(wx, 0, wz). U=(0, 1, 0).
            # Basis X' (from vl_x) = U x W = (wz, 0, -wx)
            # Basis Y' (from vl_y) = U = (0, 1, 0)
            # Basis Z' (from vl_z) = W = (wx, 0, wz)
            
            v_frag_gx = vl_x * wz + vl_y * 0.0 + vl_z * wx
            v_frag_gy = vl_x * 0.0 + vl_y * 1.0 + vl_z * 0.0
            v_frag_gz = vl_x * (-wx) + vl_y * 0.0 + vl_z * wz
            
            # IMPORTANT: Define Pilot-Centric Coordinates
            # Static data: vl_z is NOSEWARD (+Z in static frame)
            # We want Theta=0 to be TAILWARD.
            # So we bin the vector pointing TAILWARD: (vl_x, vl_y, -vl_z) -> No,
            # Let's just use the standard: bin(vl_x, vl_y, vl_z) gives Theta=0 for Noseward.
            # To make Theta=0 Tailward, we bin (-vl_x, -vl_y, -vl_z).
            theta, phi = spherical_bin(-vl_x, -vl_y, -vl_z)
            
            # v_total is the scalar speed relative to Earth, used for distance calc
            vf_x = v_frag_gx + vb_x
            vf_y = v_frag_gy + vb_y
            vf_z = v_frag_gz + vb_z
            v_total = math.sqrt(vf_x*vf_x + vf_y*vf_y + vf_z*vf_z)
            
            d_stop = calculate_d_stop(v_total, k, mass, energy_threshold)
            
            if d_stop > 0:
                # Store
                out_rays[count_idx, 0] = theta
                out_rays[count_idx, 1] = phi
                out_rays[count_idx, 2] = d_stop
                out_rays[count_idx, 3] = weight
                count_idx += 1
                

                
    return out_rays[:count_idx]

@njit(fastmath=True)
def spherical_to_cartesian(theta, phi):
    """
    Local Sphere to Local Cartesian (Standard Physics Convention)
    Theta: 0 to pi (From +Z)
    Phi: -pi to pi (From +X on XY plane)
    """
    sin_t = math.sin(theta)
    x = sin_t * math.cos(phi)
    y = sin_t * math.sin(phi)
    z = math.cos(theta)
    return x, y, z

@njit(fastmath=True)
def get_projected_area(theta_bin, phi_bin, dive_angle_rad, top, side, front):
    """
    Calculates the projected area of the aircraft based on the ray's incoming global vector.
    
    Coordinate Logic:
    1. The 'Bin' logic in generate_rays_jit produces (theta, phi) relative to the bomb's velocity vector 
       (or strictly speaking, a frame aligned with bomb's velocity W).
       - Theta=0 is TAILWARD (if we used -vl_z). 
       - Wait, generate_rays_jit does: theta, phi = spherical_bin(-vl_x, -vl_y, -vl_z).
       - vl_z was ALONG bomb velocity. So -vl_z is TAILWARD (-W).
       - So Theta=0 is TAIL direction.
    
    2. We need to reconstruct the "Global Vector" v_global of this ray.
       - Let's reconstruct vector v_tail_frame in the "Tail-Aligned Frame".
       - This frame has Z_tail = -W (Tailward), Y_tail = U (Horizontal), X_tail is derived.
       - Actually, generate_rays_jit uses (X', Y', Z') = (U x W, U, W).
       - And inputs (-vl_x, -vl_y, -vl_z).
       - So the ray vector V = (-vl_x)*X' + (-vl_y)*Y' + (-vl_z)*Z'.
       - V = v_tail_frame.x * X' + v_tail_frame.y * Y' + v_tail_frame.z * Z'
       - Wait, the bin (theta, phi) comes from spherical_bin(vx, vy, vz).
       - So vx = sin(t)cos(p), vy = sin(t)sin(p), vz = cos(t).
       - These (vx, vy, vz) ARE (-vl_x, -vl_y, -vl_z).
       - So V_global = vx * X' + vy * Y' + vz * Z'.
    
    3. Global Basis (X', Y', Z') in Earth Coordinates (East, North, Up):
       - Bomb Velocity W (Magnitude 1) is diving at 'dive_angle'.
       - Assuming standard dive: Heading 'North' (Y+) or just 'Forward'? 
       - Let's assume Bomb Heading is along X axis? No, strictly 2D X-Z in generate_rays_jit.
       - In generate_rays_jit: vb_x = cos(rad), vb_z = -sin(rad). (Down is -Z).
       - So W = (cos(ang), 0, -sin(ang)).
       - Y' = U = (0, 1, 0).
       - X' = U x W = (1*(-sin) - 0, 0, 0 - 1*cos) = (-sin(ang), 0, -cos(ang)) ???
       - Let's recheck Cross Product U(0,1,0) x W(c, 0, -s):
         x: 1*(-s) - 0 = -s
         y: 0 - 0 = 0
         z: 0 - 1*(c) = -c
       - So X' = (-sin(ang), 0, -cos(ang)).
       - NOTE: X' is "Down-Left"? Or just orthogonal.
       
    4. We need Z-component of V_global (Vertical Factor).
       - V_global_z = vx * X'_z + vy * Y'_z + vz * Z'_z
       - X'_z = -cos(dive_angle)
       - Y'_z = 0
       - Z'_z = -sin(dive_angle)
       
    5. Calculate Vertical Factor:
       - vert_factor = abs(vx * (-cos(ang)) + vz * (-sin(ang)))
    """
    
    # 1. Reconstruct local components from Bin (Theta, Phi)
    vx, vy, vz = spherical_to_cartesian(theta_bin, phi_bin)
    
    # 2. Get Vertical Component magnitude
    # Using derived dot product logic
    cos_dive = math.cos(dive_angle_rad)
    sin_dive = math.sin(dive_angle_rad)
    
    # Vz_global = vx * (-cos_dive) + vz * (-sin_dive)
    global_z = vx * (-cos_dive) + vz * (-sin_dive)
    
    vertical_factor = abs(global_z)
    
    # 3. Horizontal Factor
    # Ideally sqrt(1 - v^2), but safeguard against float error > 1
    if vertical_factor >= 1.0:
        horizontal_factor = 0.0
    else:
        horizontal_factor = math.sqrt(1.0 - vertical_factor * vertical_factor)
        
    # 4. Weighted Area
    # Top/Bottom sees vertical rays. Side/Front sees horizontal.
    # User Spec: "Vertical Factor contributes to Top. Horizontal contributes to Side."
    # Front is ignored/merged into side for now as per spec 
    # (But we passed it in just in case logic expands)
    
    # Refined Logic:
    # "Side" usually means lateral. "Front" means nose/tail.
    # Our horizontal_factor mixes Side and Front.
    # For a Box model, Horizontal Area = A_front * |V_long| + A_side * |V_lat|?
    # But User Spec says simplistically: Side (~12) vs Front (~4). Usually use Side for horizontal.
    # Let's stick to the Spec: "Horizontal component weight -> Side".
    
    area = top * vertical_factor + side * horizontal_factor
    
    # Safety Clamp
    min_area = min(top, min(side, front))
    if area < min_area:
        area = min_area
        
    return area



def solve_single_case(v_bomb, angle_bomb, fragments_data, config, density_const=DEFAULT_RHO_FRAG):
    # Unpack Config
    c_bomb = config['bomb']
    c_tgt = config['target']
    c_cmp = config['compute']
    
    n_azimuth = 144 # Increased for better coverage with 72 bins
    
    rays = generate_rays_jit(
        fragments_data,
        float(v_bomb), float(angle_bomb),
        float(c_bomb['shape_factor']), float(c_bomb['drag_coeff']), 
        float(c_bomb['air_density']), float(density_const),
        float(c_tgt['energy_threshold']),
        int(n_azimuth)
    )
    
    # Grid Aggregation
    n_theta = int(c_cmp['spatial_bins'])
    n_phi = int(c_cmp['spatial_bins']) # Assuming same resolution
    
    # Bins
    # Theta: 0 to Pi
    # Phi: -Pi to Pi
    
    # We use a simple dictionary or list-of-lists approach for sparsity?
    # Or flattened array sort?
    # Flattened Sort is best.
    # Convert theta/phi to bin index.
    
    ray_theta = rays[:, 0]
    ray_phi = rays[:, 1]
    ray_d = rays[:, 2]
    ray_w = rays[:, 3]
    
    t_idxs = (ray_theta / np.pi * n_theta).astype(np.int32)
    p_idxs = ((ray_phi + np.pi) / (2 * np.pi) * n_phi).astype(np.int32)
    
    # Clip
    t_idxs = np.clip(t_idxs, 0, n_theta - 1)
    p_idxs = np.clip(p_idxs, 0, n_phi - 1)
    
    # Flat Bin ID
    bin_ids = t_idxs * n_phi + p_idxs
    
    # Sort by Bin ID, then D_stop (descending)
    # Negate D to sort descending
    sort_keys = np.lexsort((-ray_d, bin_ids))
    
    sorted_d = ray_d[sort_keys]
    sorted_w = ray_w[sort_keys]
    sorted_bins = bin_ids[sort_keys]
    
    # Iterate Bins
    # We can use np.unique to find boundaries, but just loop is fine in Python
    # Since we need "Max Safe R" per bin, then Max over all bins?
    
    # For Envelope, we eventually want the Surface R(theta, phi).
    # Return 2D grid.
    
    envelope_r = np.zeros(n_theta * n_phi)
    
    # Calc C factor
    # dOmega approx = 4pi / (N*N)? No.
    # dOmega depends on Theta. dOmega = dTheta * dPhi * sin(Theta).
    d_theta = np.pi / n_theta
    d_phi = 2 * np.pi / n_phi
    
    # Iterate unique bins
    unique_bins, start_indices = np.unique(sorted_bins, return_index=True)
    
    # Precompute sin(theta) for bins to get area
    # Map flat bin to theta idx
    bin_t_idxs = unique_bins // n_phi
    bin_centers_theta = (bin_t_idxs + 0.5) * d_theta
    bin_omegas = d_theta * d_phi * np.sin(bin_centers_theta)
    
    # Extract safe_prob
    safe_prob = c_tgt['safe_prob']
    
    # Unpack Box Model Config
    use_box = False
    a_top = 25.0
    a_side = 12.0
    a_front = 4.0
    
    if 'area_box_model' in c_tgt and c_tgt['area_box_model'].get('enabled', False):
        use_box = True
        box_cfg = c_tgt['area_box_model']
        a_top = float(box_cfg['top'])
        a_side = float(box_cfg['side'])
        a_front = float(box_cfg['front'])
    
    # Fallback to legacy if not enabled (though enabled in recent config)
    # But actually c_tgt['area'] might be gone.
    # If calculate_projected_area is used, we calculate area per bin.
    
    # Helper for "Safe R" logic
    # C_val = SafeProb * dOmega / TargetArea -> NOW DYNAMIC
    
    # Need Dive Angle in Radians for projection
    dive_rad = math.radians(float(angle_bomb))

    for i, b_idx in enumerate(unique_bins):
        start = start_indices[i]
        end = start_indices[i+1] if i+1 < len(start_indices) else len(sorted_bins)
        
        # Rays in this bin
        d_vals = sorted_d[start:end] # Descending
        w_vals = sorted_w[start:end]
        
        # --- NEW DYNAMIC AREA CALCULATION ---
        if use_box:
            # Reconstruct Bin Center (Theta, Phi)
            t_idx = b_idx // n_phi
            p_idx = b_idx % n_phi
            
            cen_theta = (t_idx + 0.5) * d_theta
            cen_phi = (p_idx + 0.5) * d_phi - np.pi 
            
            current_area = get_projected_area(cen_theta, cen_phi, dive_rad, a_top, a_side, a_front)
        else:
            current_area = a_top
            
        c_val = (safe_prob * bin_omegas[i]) / current_area
        if c_val <= 0: c_val = 1e-9
        
        cum_w = 0.0
        max_safe_r = 0.0
        
        for k in range(len(d_vals)):
            cum_w += w_vals[k]
            limit = np.sqrt(cum_w / c_val)
            d_curr = d_vals[k]
            
            boundary = min(d_curr, limit)
            if boundary > max_safe_r:
                max_safe_r = boundary
                
        envelope_r[b_idx] = max_safe_r
        
    return envelope_r.reshape(n_theta, n_phi)


import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_3d_envelope(envelope, output_path):
    # Envelope is n_theta x n_phi grid
    n_theta, n_phi = envelope.shape
    
    # Generate Coordinates
    # Theta: 0 to pi
    # Phi: -pi to pi
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # R values
    R = envelope
    
    # Convert to Cartesian
    # X = R sin(th) cos(ph)
    # Y = R sin(th) sin(ph)
    # Z = R cos(th)
    
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    # Orientation:
    # Z is Up. X is Forward?
    # In Solver: v_bomb aligned with -Z (Dive 90) or tilted.
    # We aggregated Max R in "Global Frame".
    # Global Frame: Z Up.
    
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.9)])
    
    fig.update_layout(
        title='Mk82 Safe Escape Envelope (3D)',
        scene=dict(
            xaxis_title='X (meters) - East',
            yaxis_title='Y (meters) - North',
            zaxis_title='Z (meters) - Altitude',
            aspectmode='data'
        )
    )
    
    fig.write_html(output_path)
    print(f"Saved 3D Plot to {output_path}")

def plot_2d_slices(envelope, output_dir):
    n_theta, n_phi = envelope.shape
    
    # Slice 1: Side View (XZ Plane, Y=0, Phi=0 and Phi=Pi)
    # Phi index for 0:
    # Phi goes -pi to pi.
    # Index for 0 is roughly Mid.
    mid_phi = n_phi // 2 # Phi ~ 0
    opp_phi = 0 # Phi ~ -pi
    
    theta_vals = np.linspace(0, np.pi, n_theta)
    
    r_front = envelope[:, mid_phi] # Phi=0 (Forward/Right?)
    r_back = envelope[:, opp_phi] # Phi=-Pi (Back/Left?)
    
    # Convert to XZ coords
    # Phi=0: X = R sin(th), Z = R cos(th)
    # Phi=Pi: X = R sin(th) * (-1), Z = R cos(th)
    
    x1 = r_front * np.sin(theta_vals)
    z1 = r_front * np.cos(theta_vals)
    
    x2 = r_back * np.sin(theta_vals) * (-1)
    z2 = r_back * np.cos(theta_vals)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x1, z1, label='Azimuth 0 deg')
    plt.plot(x2, z2, label='Azimuth 180 deg')
    
    # Draw Bomb (Origin)
    plt.scatter([0], [0], c='red', marker='x', label='Release Point')
    
    plt.title("Vertical Cross Section (Side View)")
    plt.xlabel("Range (m)")
    plt.ylabel("Altitude Relative to Release (m)")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    path = os.path.join(output_dir, "envelope_side_view.png")
    plt.savefig(path)
    print(f"Saved Side View to {path}")
    plt.close()


def export_to_obj(envelope, output_path):
    """
    Exports the envelope to a Wavefront OBJ file.
    User can open this in Windows 3D Viewer, Blender, etc.
    """
    n_theta, n_phi = envelope.shape
    
    # Vertices
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    X = envelope * np.sin(THETA) * np.cos(PHI)
    Y = envelope * np.sin(THETA) * np.sin(PHI)
    Z = envelope * np.cos(THETA)
    
    points = []
    # Flatten grid to vertex list
    # Vertex index = i * n_phi + j + 1 (1-indexed)
    
    with open(output_path, 'w') as f:
        f.write("# Mk82 Safety Envelope\n")
        f.write(f"o envelope\n")
        
        # Write Vertices
        for i in range(n_theta):
            for j in range(n_phi):
                f.write(f"v {X[i,j]:.4f} {Z[i,j]:.4f} {-Y[i,j]:.4f}\n") 
                # Swapping Y/Z for common 3D viewers (Y-up) vs Engineering Z-up
                
        # Write Faces (Quads)
        for i in range(n_theta - 1):
            for j in range(n_phi - 1):
                # Indices (1-based)
                # p1 -- p2
                # |      |
                # p3 -- p4
                
                rows = n_phi
                p1 = i * rows + j + 1
                p2 = i * rows + (j + 1) + 1
                p3 = (i + 1) * rows + j + 1
                p4 = (i + 1) * rows + (j + 1) + 1
                
                f.write(f"f {p1} {p3} {p4} {p2}\n")
                
    print(f"Saved 3D Model to {output_path}")

def plot_engineering_views(envelope, output_path):
    """
    Standard 4-view engineering plot: Iso, Top, Side, Front.
    White background style.
    """
    n_theta, n_phi = envelope.shape
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    X = envelope * np.sin(THETA) * np.cos(PHI)
    Y = envelope * np.sin(THETA) * np.sin(PHI)
    Z = envelope * np.cos(THETA)
    
    fig = plt.figure(figsize=(16, 12))
    # plt.style.use('dark_background') # Removed
    plt.style.use('default') 
    
    # Common Grid Style
    grid_style = dict(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # 1. Isometric (3D)
    ax_iso = fig.add_subplot(2, 2, 1, projection='3d')
    # Use a colormap providing contrast on white, e.g., 'jet' or 'viridis' is fine
    # Add a wireframe or edges for better definition
    surf = ax_iso.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, edgecolor='k', linewidth=0.1)
    ax_iso.set_title("Isometric View", fontsize=12, fontweight='bold')
    ax_iso.set_xlabel("X (East)")
    ax_iso.set_ylabel("Y (North)")
    ax_iso.set_zlabel("Z (Alt)")
    # ax_iso.grid(**grid_style) # 3D axes have their own grid logic
    
    # 2. Top View (XY Plane)
    ax_top = fig.add_subplot(2, 2, 2)
    
    # Contour plot with labels
    cp = ax_top.contourf(X, Y, Z, 20, cmap='jet')
    plt.colorbar(cp, ax=ax_top, label='Altitude (Z)')
    
    mid_theta = n_theta // 2
    x_horiz = X[mid_theta, :]
    y_horiz = Y[mid_theta, :]
    ax_top.plot(x_horiz, y_horiz, 'b-', linewidth=2, label='Horizontal Slice')
    
    ax_top.set_title("Top View (Horizontal Projection)", fontsize=12, fontweight='bold')
    ax_top.set_aspect('equal')
    ax_top.grid(**grid_style)
    ax_top.text(0.05, 0.95, "Color = Altitude", transform=ax_top.transAxes, verticalalignment='top')
    
    # 3. Side View (XZ Plane, Y=0)
    ax_side = fig.add_subplot(2, 2, 3)
    # Phi = 0 and Phi = pi
    mid_phi = n_phi // 2 # Phi ~ 0
    opp_phi = 0 # Phi ~ -pi
    
    # Use Dark colors: Blue and Red
    ax_side.plot(X[:, mid_phi], Z[:, mid_phi], 'b-', linewidth=1.5, label='Heading 0')
    ax_side.plot(X[:, opp_phi], Z[:, opp_phi], 'r-', linewidth=1.5, label='Heading 180')
    
    # Add Reference Plane
    ax_side.scatter([0], [0], color='k', marker='x', s=100, label='Release Point')
    
    ax_side.set_title("Side View (XZ Plane)", fontsize=12, fontweight='bold')
    ax_side.set_xlabel("Range (m)")
    ax_side.set_ylabel("Altitude (m)")
    ax_side.legend(loc='upper right')
    ax_side.set_aspect('equal')
    ax_side.grid(**grid_style)

    # 4. Front View (YZ Plane, X=0) 
    ax_front = fig.add_subplot(2, 2, 4)
    # Theta varies, Phi = -pi/2 and pi/2?
    phi_90 = int(n_phi * 0.75) # pi/2
    phi_270 = int(n_phi * 0.25) # -pi/2
    
    ax_front.plot(Y[:, phi_90], Z[:, phi_90], 'b-', linewidth=1.5, label='Right (90)')
    ax_front.plot(Y[:, phi_270], Z[:, phi_270], 'r-', linewidth=1.5, label='Left (-90)')
    
    # Draw scale
    max_d = np.max(envelope)
    ax_front.add_patch(plt.Circle((0,0), 100, color='k', fill=False, linestyle='--', label='100m Ref'))
    
    ax_front.set_title("Front View (YZ Plane)", fontsize=12, fontweight='bold')
    ax_front.set_xlabel("Lateral Range (m)")
    ax_front.set_ylabel("Altitude (m)")
    ax_front.legend(loc='upper right')
    ax_front.set_aspect('equal')
    ax_front.grid(**grid_style)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white')
    print(f"Saved Engineering Plot to {output_path}")

def main():
    config_path = "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    data_path = os.path.join(config['data']['output_dir'], "envelope_result.npy")
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    envelope = np.load(data_path)
    output_dir = config['data']['output_dir']
    
    # 1. Interactive 3D Plot (HTML)
    plot_3d_envelope(envelope, os.path.join(output_dir, "envelope_3d.html"))
    
    # 2. Engineering 4-View (PNG)
    plot_engineering_views(envelope, os.path.join(output_dir, "envelope_engineering.png"))
    
    # 3. 2D Slices (PNG)
    plot_2d_slices(envelope, output_dir)
    
    print(f"Max Safe Distance: {np.max(envelope):.2f} m")

if __name__ == "__main__":
    main()

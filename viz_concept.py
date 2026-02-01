import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os

def draw_concept_diagram(output_path="data_cache/concept_diagram.png"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- 1. 定义几何参数 ---
    # 炸弹落角 (85度, 指向右下方)
    impact_angle_deg = -85.0
    impact_rad = np.radians(impact_angle_deg)
    
    # 炸弹速度矢量 (画长一点方便看)
    v_mag = 400 
    vx = v_mag * np.cos(impact_rad)
    vy = v_mag * np.sin(impact_rad)
    
    # 尾部方向 (Tail Aspect) = 速度矢量的反方向
    tail_angle_rad = impact_rad + np.pi
    
    # 安全距离 (算出来的 620m)
    safe_dist = 620.33
    
    # --- 2. 绘图元素 ---
    
    # A. 画地面 (Ground)
    ax.axhline(0, color='black', linewidth=2, linestyle='-', label='Ground Level')
    ax.text(300, -20, 'Ground', fontsize=12, va='top')
    
    # B. 画炸弹速度矢量 (Bomb Velocity Vector)
    ax.arrow(0, 0, vx, vy, head_width=20, head_length=30, fc='red', ec='red', zorder=5)
    ax.text(vx, vy, f'  Bomb Velocity\n  (260 m/s @ {abs(impact_angle_deg)}° Dive)', 
            color='red', fontsize=12, va='top')
    
    # C. 画尾部安全扇区 (Tail Safe Corridor)
    # 扇区范围: 尾部轴线 +/- 15度
    wedge_radius = 800
    theta1 = np.degrees(tail_angle_rad) - 15
    theta2 = np.degrees(tail_angle_rad) + 15
    
    # 绘制扇区 (半透明绿)
    wedge = patches.Wedge((0, 0), wedge_radius, theta1, theta2, 
                          color='green', alpha=0.2, label='Safe Sector (<15° Tail)')
    ax.add_patch(wedge)
    
    # 画尾部中心轴线 (Reference Line theta=0 in simulation)
    ax.plot([0, wedge_radius * np.cos(tail_angle_rad)], 
            [0, wedge_radius * np.sin(tail_angle_rad)], 
            color='green', linestyle='--', linewidth=1.5)
    ax.text(wedge_radius * np.cos(tail_angle_rad), wedge_radius * np.sin(tail_angle_rad), 
            '  Tail Axis (Theta=0)', color='green', fontsize=12)

    # D. 画载机位置 (Aircraft)
    # 假设飞机刚好在安全距离边缘，且位于正尾部
    ac_x = safe_dist * np.cos(tail_angle_rad)
    ac_y = safe_dist * np.sin(tail_angle_rad)
    
    # 画飞机图标 (用简单的三角形代替)
    # 飞机朝向: 水平向右飞 (假设平飞投弹)
    plane_marker = dict(marker='>', markersize=15, color='blue', zorder=10, label='Aircraft')
    ax.plot(ac_x, ac_y, **plane_marker)
    
    # E. 画斜距连线 (Slant Range Line) - 核心重点！
    ax.plot([0, ac_x], [0, ac_y], color='blue', linestyle='-', linewidth=2)
    
    # 标注斜距数值
    mid_x = ac_x / 2
    mid_y = ac_y / 2
    ax.annotate(f'Safe Slant Range\nR = {safe_dist:.1f} m', 
                xy=(mid_x, mid_y), xytext=(mid_x+100, mid_y),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                fontsize=12, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

    # --- 3. 图表修饰 ---
    ax.set_aspect('equal')
    ax.set_title(f'Mk82 Snakeye Safe Separation Geometry\n(Tail Initiation / High Drag)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance Downrange (m)')
    ax.set_ylabel('Altitude (m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    
    # 设置显示范围
    ax.set_xlim(-200, 800)
    ax.set_ylim(-300, 800)
    
    # 原点爆炸点
    ax.scatter([0], [0], color='orange', s=200, marker='*', zorder=6, label='Burst Point')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Schematic saved to {output_path}")

if __name__ == "__main__":
    draw_concept_diagram()

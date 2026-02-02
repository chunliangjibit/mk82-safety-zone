import numpy as np
import yaml
import os
import math
import datetime

def calculate_dynamic_area(sin_alpha, top, side):
    """
    重现 core_solver 中的面积计算逻辑，用于报告展示。
    sin_alpha: 视线仰角的正弦值 (Vertical Factor)
    """
    vertical_factor = abs(sin_alpha)
    # 水平因子 (Horizontal Factor)
    if vertical_factor >= 1.0:
        horizontal_factor = 0.0
    else:
        horizontal_factor = math.sqrt(1.0 - vertical_factor**2)
    
    area = top * vertical_factor + side * horizontal_factor
    return area

def scan_envelope_metrics(envelope, dive_angle_deg, a_top, a_side):
    """
    全数据网格扫描分析器。
    将每个网格点还原为 (SlantRange, Height, Area)，并进行统计分析。
    """
    n_theta, n_phi = envelope.shape
    dive_rad = math.radians(dive_angle_deg)
    
    # 结果容器
    # 格式: [SlantRange, Height, Area]
    all_points = []
    
    d_theta = np.pi / n_theta
    d_phi = 2 * np.pi / n_phi
    
    for i in range(n_theta):
        # 还原局部 Theta
        theta_local = (i + 0.5) * d_theta
        
        # 预计算局部 Z 分量 (Local Vz = cos(theta))
        local_vz = math.cos(theta_local)
        local_rho = math.sin(theta_local) # sin(theta)
        
        for j in range(n_phi):
            # 还原局部 Phi
            phi_local = (j + 0.5) * d_phi - np.pi
            
            # 读取斜距 R
            r = envelope[i, j]
            if r <= 1.0: continue # 忽略无效点
            
            # --- 坐标变换: 局部 -> 全局 Z (高度) ---
            # Local Cartesian (Normalized)
            lx = local_rho * math.cos(phi_local)
            # ly = local_rho * math.sin(phi_local) # Y不影响高度
            lz = local_vz
            
            # Global Z (Height) calculation
            # 旋转矩阵 (针对俯冲角 Dive):
            # Height = -(lx * cos(Dive) + lz * sin(Dive)) * R
            # 注意: 炸弹向下俯冲，所以 Global Z 向上为负方向的投影取反?
            # 让我们回顾 core_solver: vb_x = cos(rad), vb_z = -sin(rad). (Down is -Z)
            # Global Basis Z' = (wx, 0, wz) = (cos, 0, -sin)?
            # Wait, core_solver lines 105-107:
            # v_frag_gz = vl_x * (-wx) + vl_y * 0.0 + vl_z * wz
            # We need to act consistently with core_solver logic.
            # But here we are just doing geometric projection.
            # Let's trust the user's provided formula for now: 
            # h_factor = -(lx * math.cos(dive_rad) + lz * math.sin(dive_rad))
            
            h_factor = -(lx * math.cos(dive_rad) + lz * math.sin(dive_rad))
            height = r * h_factor
            
            # 只关心上半球 (Height > 0) 的点，忽略炸弹下方的点
            if height < 0: continue
            
            # --- 重算当时的暴露面积 ---
            # 视线仰角的正弦值 sin(alpha) = Height / SlantRange = h_factor
            # 我们的 Box Model 逻辑: 垂直权重 = |sin(alpha)|
            current_area = calculate_dynamic_area(h_factor, a_top, a_side)
            
            all_points.append({
                'r': r,
                'h': height,
                'area': current_area
            })
            
    return all_points

def generate_calculation_report(config_path, data_path, output_dir):
    """
    Generate the Tactical Analysis Report with comprehensive Metadata.
    """
    print("Generating Comprehensive Tactical Analysis Report...")
    
    # 1. Load Config
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    envelope = np.load(data_path)
    
    # 获取 Box Model 参数
    box_cfg = config['target']['area_box_model']
    a_top = box_cfg['top']
    a_side = box_cfg['side']
    
    # 获取俯冲角参考 (取最大俯冲角作为最严酷几何条件)
    dive_ref = config['scan']['angle']['max']
    
    # --- 核心分析步骤 ---
    points = scan_envelope_metrics(envelope, dive_ref, a_top, a_side)
    
    if not points:
        print("Error: No valid data points found.")
        return

    # [分析 1] 全空域最大值 (Global Max)
    global_max_point = max(points, key=lambda p: p['r'])
    
    # [分析 2] 战术高度层最大值 (Tactical Band Max)
    if 'tactical_band' in config['target']:
        band_min = config['target']['tactical_band']['min_height']
        band_max = config['target']['tactical_band']['max_height']
    else:
        band_min = 50.0
        band_max = 150.0
    
    band_points = [p for p in points if band_min <= p['h'] <= band_max]
    tactical_point = max(band_points, key=lambda p: p['r']) if band_points else {'r': 0.0, 'h': 0.0, 'area': 0.0}

    # Generate Path
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"envelope_report_{ts}.txt"
    report_path = os.path.join(output_dir, report_filename)

    # --- 生成报告文本 ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========================================================\n")
        f.write("       战术安全包络线分析报告 (Tactical Safety Report)      \n")
        f.write("========================================================\n\n")
        
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"数据文件: {data_path}\n\n")

        f.write("--------------------------------------------------------\n")
        f.write("1. 基础元数据 (Input Metadata)\n")
        f.write("--------------------------------------------------------\n")
        
        c = config['bomb']
        f.write("[炸弹物理参数]\n")
        f.write(f"  质量 (Mass):         {c['mass']} kg\n")
        f.write(f"  形状系数 (Shape):    {c['shape_factor']}\n")
        f.write(f"  阻力系数 (Drag):     {c['drag_coeff']}\n")
        f.write(f"  空气密度 (Air):      {c['air_density']} kg/m^3\n\n")
        
        t = config['target']
        f.write("[目标与安全判据]\n")
        f.write(f"  动态盒模型 (Box Model): 已启用\n")
        f.write(f"    - 机腹/机背投影: {a_top} m^2\n")
        f.write(f"    - 侧面投影:     {a_side} m^2\n")
        f.write(f"    - 迎头/尾部投影: {box_cfg.get('front', 4.0)} m^2\n")
        f.write(f"  安全概率阈值:    {t['safe_prob']}\n")
        f.write(f"  致死动能门限:    {t['energy_threshold']} J\n\n")
        
        s = config['scan']
        f.write("[扫描范围]\n")
        f.write(f"  落速 (m/s): {s['velocity']['min']} - {s['velocity']['max']} (步长: {s['velocity']['step']})\n")
        f.write(f"  落角 (deg): {s['angle']['min']} - {s['angle']['max']} (步长: {s['angle']['step']})\n\n")

        f.write("--------------------------------------------------------\n")
        f.write("2. 全空域平飞极值分析 (Global Level-Flight Peak)\n")
        f.write("--------------------------------------------------------\n")
        f.write("   [定义]: 假设飞机保持平飞姿态，扫描所有高度/方位，找到的最坏情况。\n")
        f.write(f"   >>> 最大安全距离: {global_max_point['r']:.2f} 米 <<<\n\n")
        f.write(f"   [关键状态回溯]:\n")
        f.write(f"     - 发生高度 (Height):      {global_max_point['h']:.2f} 米\n")
        # Calc angle carefully to avoid domain error
        ratio = min(1.0, max(-1.0, global_max_point['h']/global_max_point['r']))
        look_ang = math.degrees(math.asin(ratio))
        f.write(f"     - 此时视线仰角 (Look Ang): {look_ang:.1f} 度\n")
        f.write(f"     - 此时暴露面积 (Area):    {global_max_point['area']:.2f} m^2\n\n")
        f.write(f"   [解读]: 如果仰角接近 90 度，说明危险源来自正下方载机暴露了最大机腹面积 ({a_top}m2)。\n\n")

        f.write("--------------------------------------------------------\n")
        f.write("3. 战术高度层推荐 (Tactical Band Recommendation)\n")
        f.write("--------------------------------------------------------\n")
        f.write(f"   [预设战术条件]:\n")
        f.write(f"     - 飞行高度层: {band_min}m 至 {band_max}m\n")
        f.write(f"     - 飞行姿态:   保持平飞 (Wings Level)\n\n")
        f.write(f"   >>> 战术推荐距离: {tactical_point['r']:.2f} 米 <<<\n\n")
        f.write(f"   [关键状态回溯]:\n")
        f.write(f"     - 发生高度 (Height):      {tactical_point['h']:.2f} 米\n")
        f.write(f"     - 此时暴露面积 (Area):    {tactical_point['area']:.2f} m^2\n\n")
        f.write(f"   [解读]: 此距离是该高度层内，综合最坏来袭角度与实际暴露面积后的安全红线。\n\n")
        f.write("========================================================\n")
        f.write("报告结束 (END OF REPORT)\n")
        f.write("========================================================\n")
    
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    generate_calculation_report("config.yaml", "data_cache/envelope_result.npy", "data_cache")

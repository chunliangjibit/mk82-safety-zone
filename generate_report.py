
import yaml
import os
import datetime
import numpy as np

def generate_calculation_report(config_path, data_path, output_dir):
    """
    Generates a detailed text report of the calculation in Chinese.
    """
    # Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Load Results
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return
        
    envelope = np.load(data_path)
    
    # --- Data Analysis for Report ---
    # Envelope is 2D Grid: Theta (Rows) x Phi (Cols)
    # Theta: 0 to Pi (0=Up, Pi=Down, Pi/2=Horizon)
    params = config['compute']
    n_theta = int(params['spatial_bins'])
    
    # 1. Global Max (Omnidirectional limit)
    max_safe_dist = np.max(envelope)
    
    # 2. Vertical Aspect (Top/Bottom) -> Theta near 0 or Pi
    # Slices: Top 5 rows and Bottom 5 rows
    slice_width = max(1, n_theta // 10) # Top 10%
    vert_slice_top = envelope[0:slice_width, :]
    vert_slice_bot = envelope[-slice_width:, :]
    max_dist_vertical = max(np.max(vert_slice_top), np.max(vert_slice_bot))
    
    # 3. Lateral Aspect (Side/Horizon) -> Theta near Pi/2 (Index N/2)
    # Slice: Middle 10%
    mid_idx = n_theta // 2
    half_width = max(1, n_theta // 20)
    lat_slice = envelope[mid_idx-half_width : mid_idx+half_width, :]
    max_dist_lateral = np.max(lat_slice)
    
    box_enabled = False
    if 'area_box_model' in config['target'] and config['target']['area_box_model'].get('enabled', False):
        box_enabled = True
        ab = config['target']['area_box_model']
    
    # --- Generate Report ---
    
    # Generate Filename with Timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"envelope_report_{ts}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========================================================\n")
        f.write("        MK82 安全包络线计算报告 (Safety Envelope Report)      \n")
        f.write("========================================================\n\n")
        
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"数据文件: {data_path}\n\n")
        
        f.write("--------------------------------------------------------\n")
        f.write("1. 输入参数 (Input Parameters)\n")
        f.write("--------------------------------------------------------\n")
        
        c = config['bomb']
        f.write("[炸弹物理参数]\n")
        f.write(f"  质量 (Mass):         {c['mass']} kg\n")
        f.write(f"  形状系数 (Shape):    {c['shape_factor']}\n")
        f.write(f"  阻力系数 (Drag):     {c['drag_coeff']}\n")
        f.write(f"  空气密度 (Air):      {c['air_density']} kg/m^3\n\n")
        
        t = config['target']
        f.write("[目标与安全判据]\n")
        if box_enabled:
            ab = t['area_box_model']
            f.write(f"  动态盒模型 (Box Model): 已启用 (ENABLED)\n")
            f.write(f"    - 机腹/机背面积 (Top/Bottom): {ab['top']} m^2  (用于垂直威胁计算)\n")
            f.write(f"    - 侧面面积 (Side/Lateral):    {ab['side']} m^2  (用于水平威胁计算)\n")
            f.write(f"    - 迎头/尾部面积 (Front/Tail): {ab['front']} m^2\n")
        else:
            f.write(f"  固定暴露面积 (Fixed Area): {t.get('area', 'N/A')} m^2\n")
            
        f.write(f"  安全概率阈值 (Prob):    {t['safe_prob']}\n")
        f.write(f"  致死动能门限 (Energy):  {t['energy_threshold']} J\n\n")
        
        s = config['scan']
        f.write("[扫描工况]\n")
        f.write(f"  投弹速度: {s['velocity']['min']} - {s['velocity']['max']} m/s\n")
        f.write(f"  投弹俯冲角: {s['angle']['min']} - {s['angle']['max']} 度\n\n")
            
        f.write("--------------------------------------------------------\n")
        f.write("2. 计算结果 (Calculation Results)\n")
        f.write("--------------------------------------------------------\n")
        
        # 3.1 Global Max
        f.write("[A] 综合最大安全距离 (Omnidirectional Max)\n")
        f.write(f"    >>> {max_safe_dist:.2f} 米 <<<\n\n")
        f.write("    [定义]: 全空域最坏情况下的最大危险距离。\n")
        f.write("            (无论载机位于哪个方位，大于此距离即绝对安全)\n\n")
        
        # 3.2 Aspect Analysis
        f.write("[B] 分量安全距离分析 (Aspect Analysis)\n")
        f.write("    系统根据破片来袭方向，实时计算了对应的动态有效投影面积 (Effective Projected Area)：\n\n")
        
        # Vertical
        f.write(f"    1. 垂直/高抛威胁 (Vertical Hazard) - 对应机腹/机背\n")
        f.write(f"       有效暴露面积: {ab['top'] if box_enabled else t.get('area')} m^2\n")
        f.write(f"       安全距离:     {max_dist_vertical:.2f} 米\n")
        f.write(f"       [说明]: 来自正上方或正下方的破片威胁，通常决定了全向最大距离。\n")
        if box_enabled:
            f.write(f"       [战术意义]: 若执行俯冲拉起 (Dive Pull-up) 或直接穿越爆炸点上空 (Overflight)，\n")
            f.write(f"                   此时机腹/机背完全暴露于威胁中，必须严格参考此安全距离。\n")
        f.write("\n")
        
        # Lateral
        f.write(f"    2. 侧向/水平威胁 (Lateral Hazard) - 对应侧面\n")
        f.write(f"       有效暴露面积: {ab['side'] if box_enabled else t.get('area')} m^2  <-- 随着角度变化动态计算得到\n")
        f.write(f"       安全距离:     {max_dist_lateral:.2f} 米\n")
        if box_enabled:
             f.write(f"       [战术意义]: 若飞行员保持 Snakeye 水平投放或侧向规避，\n")
             f.write(f"                   可参考此距离 ({max_dist_lateral:.0f}m) 这里的风险显著低于垂直方向。\n")
             
        f.write("\n")
        f.write("--------------------------------------------------------\n")
        f.write("报告结束 (END OF REPORT)\n")
        f.write("========================================================\n")
        
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    generate_calculation_report("config.yaml", "data_cache/envelope_result.npy", "data_cache")

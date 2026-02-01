
import os
import re
import numpy as np
import yaml

# Constants
GRAINS_TO_KG = 6.479891e-5
FTS_TO_MS = 0.3048
DEG_TO_RAD = np.pi / 180.0

def parse_markdown_table(lines):
    """
    Parses a markdown table into a list of dicts.
    Assumes the first two rows are header and separator.
    """
    table_data = []
    headers = []
    
    # Find header
    start_idx = -1
    for i, line in enumerate(lines):
        if len(line.strip()) > 0: print(f"DEBUG CHECK LINE: {line[:50]}")
        if line.strip().startswith('|') and 'Polar Zone' in line:
            start_idx = i
            headers = [h.strip() for h in line.strip().split('|') if h.strip()]
            break
            
    if start_idx == -1:
        print("DEBUG: Header not found in block")
        return None
        
    # Skip separator line
    data_start = start_idx + 2
    
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line.startswith('|'):
            break
            
        cells = [c.strip() for c in line.split('|') if c.strip() != '']
        # Handle cases where trailing pipe makes split empty
        if len(cells) != len(headers):
             # Try to match loose
             pass

        if len(cells) > 0 and 'Mean' not in cells[0] and 'Total' not in cells[0]:
             row_dict = {}
             # Zone parsing "0.0 - 4.0"
             zone_str = cells[0].replace('**', '') # Remove bold
             try:
                 z_min, z_max = map(float, zone_str.split('-'))
                 row_dict['zone_min'] = z_min
                 row_dict['zone_max'] = z_max
                 row_dict['zone_mid'] = (z_min + z_max) / 2.0
             except:
                 continue

             for h, c in zip(headers[1:], cells[1:]):
                 row_dict[h] = c
             
             table_data.append(row_dict)
             
    return table_data

def process_bomb_data(lines, bomb_name):
    """
    Process all sheets for a specific bomb.
    """
    print(f"Processing {bomb_name}...")
    print(f"DEBUG Content Start: {'^'.join(lines[:20])}")
    
    # 1. Extract Velocity Data from Summary Sheet
    # The summary sheet usually has "Mean Initial Velocity"
    velocity_map = {} # MidPoint -> Velocity
    
    # Line by line state machine
    
    current_lines = []
    frag_sheets = []
    fragments = []
    
    # Identify sheet type
    # Types: "summary", "fragments", "unknown"
    sheet_type = "unknown"
    
    # Helper to process accumulated lines
    def flush_sheet(lines, s_type):
        if not lines: return
        if s_type == "summary":
            # Parsing summary works
            # print("DEBUG: Parsing Summary")
            data = parse_markdown_table(lines)
            if data:
                 # Extract velocity
                 for row in data:
                     # Find velocity key
                     vel_key = next((k for k in row.keys() if 'Velocity' in k), None)
                     if vel_key and row[vel_key] != '-':
                        try:
                             vel_val = float(str(row[vel_key]).replace(',',''))
                             velocity_map[row['zone_mid']] = vel_val * FTS_TO_MS
                        except:
                             pass
        elif s_type == "fragments":
            # print("DEBUG: Parsing Fragments")
            frag_sheets.append(parse_markdown_table(lines))

    for line in lines:
        if line.strip().startswith('#### Sheet'):
            # New sheet starting, flush old
            flush_sheet(current_lines, sheet_type)
            current_lines = []
            
            # Determine new type
            if "Polar Zone Summary" in line:
                sheet_type = "summary"
            elif "Weight Interval" in line:
                sheet_type = "fragments"
            else:
                sheet_type = "unknown"
                
            # Keep header line? parse_markdown_table expects header line?
            # parse_markdown_table looks for "| Polar Zone"
            # It ignores Lines before it. 
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Flush last sheet
    flush_sheet(current_lines, sheet_type)
    
    if not velocity_map:
        print(f"Error: No summary data found for {bomb_name}")
        # print("DEBUG: Velocity Map is empty")
        return None

    # generic function to parse M/N columns
    # Header format: "0.0-1.0 (M)" or "0.0-1.0 (N)"
    
    for sheet_data in frag_sheets:
        if not sheet_data: continue
        
        for row in sheet_data:
            zone_mid = row['zone_mid']
            
            # Get velocity for this zone
            # Handle float precision issues
            matched_vel = None
            for v_zone, v_val in velocity_map.items():
                if abs(v_zone - zone_mid) < 0.1:
                    matched_vel = v_val
                    break
            
            if matched_vel is None:
                # Fallback or Skip? 
                # Some zones might have 0 fragments so velocity undefined?
                # Check summary table for that zone.
                # If checking C32 sheet 1, zone 4.0-8.0 has fragments, but summary says...
                # Actually summary table has all zones.
                print(f"Warning: No velocity for Zone {zone_mid} in {bomb_name}")
                continue
                
            # Iterate through columns to find pairs of (M) and (N)
            # Keys look like "0.0-1.0 (M)"
            keys = list(row.keys())
            mass_keys = [k for k in keys if '(M)' in k]
            
            for mk in mass_keys:
                nk = mk.replace('(M)', '(N)')
                
                m_str = row.get(mk, '-')
                n_str = row.get(nk, '-')
                
                if m_str != '-' and n_str != '-':
                    try:
                        mass_grains = float(m_str.replace(',',''))
                        count = float(n_str.replace(',',''))
                        
                        if count > 0:
                            fragments.append([
                                mass_grains * GRAINS_TO_KG, # Mass
                                count,                      # Count
                                matched_vel,                # Velocity (Static)
                                zone_mid * DEG_TO_RAD       # Angle
                            ])
                    except ValueError:
                        pass
                        
    return np.array(fragments, dtype=np.float64)

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("Config not found")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    source_file = config['data']['source_file']
    output_dir = config['data']['output_dir']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split C32 and C33
    # Look for "### Table C-32" and "### Table C-33"
    
    # Split C32 and C33 using regex which is robust
    # Look for "### Table C-32" and "### Table C-33"
    
    # We use capturing group to keep the delimiter to know which is which? 
    # Or just assume order. C-32 is first.
    
    sections = re.split(r'(### Table C-\d+.*)', content)
    # Result: [Preamble, Header1, Content1, Header2, Content2]
    
    c32_content = None
    c33_content = None
    
    # Iterate Sections to find them
    for i in range(len(sections)-1):
        if "Table C-32" in sections[i]:
            c32_content = sections[i] + "\n" + sections[i+1] # Include Header? 
            # Actually process_bomb_data doesn't care about the Main Header, only Sheets
            c32_content = sections[i+1]
        elif "Table C-33" in sections[i]:
            c33_content = sections[i+1]
            
    if c32_content:
        data_c32 = process_bomb_data(c32_content.split('\n'), "Mk82 (Tritonal)")
        if data_c32 is not None and len(data_c32) > 0:
            out_path = os.path.join(output_dir, config['data']['mk82_file'])
            np.save(out_path, data_c32)
            print(f"Saved {len(data_c32)} fragment groups to {out_path}")
            
    if c33_content:
        data_c33 = process_bomb_data(c33_content.split('\n'), "BLU-111 (PBXN-109)")
        if data_c33 is not None and len(data_c33) > 0:
            out_path = os.path.join(output_dir, config['data']['blu111_file'])
            np.save(out_path, data_c33)
            print(f"Saved {len(data_c33)} fragment groups to {out_path}")

if __name__ == "__main__":
    main()

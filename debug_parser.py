#!/usr/bin/env python3
"""
Debug script to understand the well connections file format
"""

def debug_well_connections():
    filepath = "/workspace/HM/HM_WELL_CONNECTIONS.ixf"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print("=== File Analysis ===")
    print(f"Total lines: {len(lines)}")
    
    # Find all WellDef sections
    well_sections = []
    for i, line in enumerate(lines):
        if line.strip().startswith('WellDef'):
            well_name = line.strip().split('"')[1]
            well_sections.append((i, well_name))
    
    print(f"Found wells: {[name for _, name in well_sections]}")
    
    # Analyze first well in detail
    if well_sections:
        start_line, well_name = well_sections[0]
        print(f"\n=== Analyzing {well_name} (starting at line {start_line+1}) ===")
        
        # Show 20 lines starting from the well definition
        for i in range(start_line, min(start_line + 30, len(lines))):
            line = lines[i].rstrip()
            print(f"{i+1:3d}: {line}")
            
            # Look for connection data pattern
            if line.strip().startswith('(') and ')' in line:
                parts = line.strip().split()
                print(f"     -> Found connection with {len(parts)} parts")
                if len(parts) >= 11:
                    coord_part = parts[0]
                    print(f"     -> Coordinate part: '{coord_part}'")

if __name__ == "__main__":
    debug_well_connections()
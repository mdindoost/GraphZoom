#!/usr/bin/env python3
"""
Debug MTX file format to understand what GraphZoom creates
"""

import sys
import os

def examine_mtx_file(filepath):
    """Examine MTX file format"""
    print(f"🔍 Examining MTX file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist: {filepath}")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"📊 File size: {file_size} bytes")
    
    # Read and analyze the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"📄 Total lines: {len(lines)}")
    
    # Show first 20 lines
    print(f"\n📋 First 20 lines:")
    for i, line in enumerate(lines[:20]):
        print(f"  {i+1:3d}: {line.strip()}")
    
    # Analyze the format
    print(f"\n🔍 Format Analysis:")
    
    # Find header information
    comment_lines = 0
    header_line = None
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('%'):
            comment_lines += 1
        elif header_line is None:
            header_line = line
            data_start = i + 1
            break
    
    print(f"   Comment lines: {comment_lines}")
    print(f"   Header line: {header_line}")
    print(f"   Data starts at line: {data_start}")
    
    # Analyze header
    if header_line:
        parts = header_line.split()
        print(f"   Header parts: {parts}")
        if len(parts) >= 2:
            print(f"   Matrix dimensions: {parts[0]} x {parts[1]}")
            if len(parts) >= 3:
                print(f"   Non-zeros: {parts[2]}")
    
    # Analyze data format
    if data_start < len(lines):
        print(f"\n📊 Data format analysis:")
        sample_lines = lines[data_start:data_start+10]
        
        for i, line in enumerate(sample_lines):
            line = line.strip()
            if line:
                parts = line.split()
                print(f"   Data line {i+1}: {parts} (length: {len(parts)})")
        
        # Check consistency
        data_lengths = []
        for line in lines[data_start:data_start+100]:  # Check first 100 data lines
            line = line.strip()
            if line:
                parts = line.split()
                data_lengths.append(len(parts))
        
        if data_lengths:
            unique_lengths = set(data_lengths)
            print(f"   Data line lengths: {unique_lengths}")
            print(f"   Most common length: {max(set(data_lengths), key=data_lengths.count)}")
    
    # Check for different MTX formats
    print(f"\n🎯 Format Detection:")
    
    # Check if it's coordinate format
    if header_line and len(header_line.split()) == 3:
        print("   Format: Coordinate (COO) - rows, cols, nnz")
    elif header_line and len(header_line.split()) == 2:
        print("   Format: Dense or Array - rows, cols")
    
    # Check if values are included
    if data_start < len(lines):
        sample_line = lines[data_start].strip()
        if sample_line:
            parts = sample_line.split()
            if len(parts) == 2:
                print("   Values: Pattern only (no weights)")
            elif len(parts) == 3:
                print("   Values: Weighted (with values)")
            else:
                print(f"   Values: Unknown format ({len(parts)} columns)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mtx_debug.py <mtx_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    examine_mtx_file(filepath)

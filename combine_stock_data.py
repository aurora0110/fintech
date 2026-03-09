#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine stock data from Excel file and individual stock data files
"""

import pandas as pd
import os
import glob

def extract_stock_codes_from_excel(excel_file_path):
    """Extract unique stock codes from the third column of Excel file"""
    try:
        # The file is actually a text file with ISO-8859 encoding and tab separator
        df = pd.read_csv(excel_file_path, sep='\t', header=0, encoding='iso-8859-1')
        
        # Get the third column (index 2)
        stock_codes = df.iloc[:, 2].dropna().unique()
        
        # Convert to string and remove any whitespace
        stock_codes = [str(code).strip() for code in stock_codes]
        
        # Pad with leading zeros to make 6-digit codes
        stock_codes_6digit = [code.zfill(6) for code in stock_codes]
        
        print(f"Found {len(stock_codes)} unique stock codes in Excel file")
        print(f"Original sample: {stock_codes[:5]}")
        print(f"6-digit sample: {stock_codes_6digit[:5]}")
        return stock_codes_6digit
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def find_stock_data_files(stock_codes, data_directory):
    """Find corresponding stock data files for each stock code"""
    stock_files = {}
    
    # Get all .txt files in the directory
    all_files = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
    
    for code in stock_codes:
        # Try different market prefixes
        market_prefixes = ['SH#', 'SZ#', 'BJ#']
        found = False
        
        for prefix in market_prefixes:
            file_pattern = f"{prefix}{code}.txt"
            file_path = os.path.join(data_directory, file_pattern)
            
            if os.path.exists(file_path):
                stock_files[code] = file_path
                found = True
                break
        
        if not found:
            # Try to find the file by extracting code from filename
            for filename in all_files:
                # Extract the 6-digit code from filename (SH#601336.txt -> 601336)
                if (filename.startswith('SH#') or filename.startswith('SZ#') or filename.startswith('BJ#')) and filename.endswith('.txt'):
                    file_code = filename[3:-4]  # Remove prefix and '.txt'
                    if file_code == code:
                        stock_files[code] = os.path.join(data_directory, filename)
                        found = True
                        break
        
        if not found:
            print(f"Warning: Data file not found for stock code {code}")
    
    print(f"Found {len(stock_files)} matching stock data files")
    return stock_files

def combine_stock_data(stock_files, output_file):
    """Combine stock data into a single text file with required format"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, (stock_code, file_path) in enumerate(stock_files.items()):
            try:
                # Write stock code
                outfile.write(f"{stock_code}\n")
                
                # Read and write stock data
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    outfile.write(content)
                
                # Add newline separator between stocks (except for last one)
                if i < len(stock_files) - 1:
                    outfile.write("\n\n")
                
                print(f"Processed stock code: {stock_code}")
                
            except Exception as e:
                print(f"Error processing stock code {stock_code}: {e}")

def main():
    # File paths
    excel_file = "/Users/lidongyang/Desktop/Qstrategy/data/Table.xls"
    data_directory = "/Users/lidongyang/Desktop/Qstrategy/data/20260226/normal"
    output_file = "/Users/lidongyang/Desktop/Qstrategy/combined_stock_data.txt"
    
    print("Starting stock data combination process...")
    
    # Step 1: Extract stock codes from Excel
    stock_codes = extract_stock_codes_from_excel(excel_file)
    
    if not stock_codes:
        print("No stock codes found. Exiting.")
        return
    
    # Step 2: Find corresponding data files
    stock_files = find_stock_data_files(stock_codes, data_directory)
    
    if not stock_files:
        print("No matching stock data files found. Exiting.")
        return
    
    # Step 3: Combine data into single file
    combine_stock_data(stock_files, output_file)
    
    print(f"\nCombined stock data saved to: {output_file}")
    print(f"Total stocks processed: {len(stock_files)}")

if __name__ == "__main__":
    main()
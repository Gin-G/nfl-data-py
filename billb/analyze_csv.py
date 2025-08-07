#!/usr/bin/env python3
"""
CSV Data Analyzer for NFL Data Import
Analyzes CSV files and maps them to database schema
"""

import pandas as pd
import os
from typing import Dict, List, Any
import json

def analyze_csv_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Basic info
        analysis = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "shape": df.shape,
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "row_count": len(df),
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head(3).to_dict('records'),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_values": {}
        }
        
        # Check for key columns
        key_columns = ['player_id', 'player_name', 'position', 'team', 'week', 'season', 'game_id']
        analysis["has_key_columns"] = {col: col in df.columns for col in key_columns}
        
        # Get unique values for categorical columns
        categorical_cols = ['position', 'team', 'season', 'week']
        for col in categorical_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                analysis["unique_values"][col] = {
                    "count": len(unique_vals),
                    "values": unique_vals[:20].tolist()  # First 20 values
                }
        
        # Look for statistical columns
        stat_patterns = [
            'passing', 'rushing', 'receiving', 'fantasy', 'yards', 'tds', 'attempts', 
            'completions', 'targets', 'receptions', 'interceptions', 'fumbles', 'sacks'
        ]
        
        stat_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in stat_patterns):
                stat_columns.append(col)
        
        analysis["stat_columns"] = stat_columns
        analysis["stat_column_count"] = len(stat_columns)
        
        return analysis
        
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": str(e),
            "file_path": file_path
        }

def analyze_directory(directory_path: str) -> Dict[str, Any]:
    """Analyze all CSV files in a directory"""
    results = {
        "directory": directory_path,
        "files_analyzed": 0,
        "files_with_errors": 0,
        "file_analyses": [],
        "summary": {}
    }
    
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        analysis = analyze_csv_file(file_path)
        
        results["file_analyses"].append(analysis)
        results["files_analyzed"] += 1
        
        if "error" in analysis:
            results["files_with_errors"] += 1
    
    # Create summary
    if results["file_analyses"]:
        all_columns = set()
        all_stat_columns = set()
        seasons = set()
        positions = set()
        
        for analysis in results["file_analyses"]:
            if "error" not in analysis:
                all_columns.update(analysis["columns"])
                all_stat_columns.update(analysis.get("stat_columns", []))
                
                if "unique_values" in analysis:
                    if "season" in analysis["unique_values"]:
                        seasons.update(analysis["unique_values"]["season"]["values"])
                    if "position" in analysis["unique_values"]:
                        positions.update(analysis["unique_values"]["position"]["values"])
        
        results["summary"] = {
            "total_unique_columns": len(all_columns),
            "total_stat_columns": len(all_stat_columns),
            "all_columns": sorted(list(all_columns)),
            "all_stat_columns": sorted(list(all_stat_columns)),
            "seasons_found": sorted(list(seasons)),
            "positions_found": sorted(list(positions))
        }
    
    return results

def map_to_database_schema(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Map CSV columns to database schema"""
    
    # Database PlayerStats columns
    db_columns = {
        # Core fields
        "player_id": "String",
        "game_id": "String", 
        "week": "Integer",
        "season": "Integer",
        
        # Passing stats
        "passing_yards": "Float",
        "passing_attempts": "Integer",
        "passing_completions": "Integer", 
        "passing_tds": "Integer",
        "interceptions": "Integer",
        "sack_yards": "Float",
        "sack_fumbles": "Integer",
        "sack_fumbles_lost": "Integer",
        
        # Rushing stats
        "rushing_yards": "Float",
        "rushing_attempts": "Integer",
        "rushing_tds": "Integer",
        "rushing_fumbles": "Integer",
        "rushing_fumbles_lost": "Integer",
        
        # Receiving stats  
        "receiving_yards": "Float",
        "targets": "Integer",
        "receptions": "Integer",
        "receiving_tds": "Integer",
        "receiving_fumbles": "Integer",
        "receiving_fumbles_lost": "Integer",
        
        # Fantasy/other stats
        "fantasy_points": "Float",
        "fantasy_points_ppr": "Float",
        "offensive_snaps": "Integer",
        "offensive_snap_pct": "Float"
    }
    
    mapping_results = {
        "database_columns": db_columns,
        "csv_to_db_mapping": {},
        "unmapped_csv_columns": [],
        "missing_db_columns": [],
        "mapping_suggestions": {}
    }
    
    if "summary" in analysis:
        csv_columns = analysis["summary"]["all_columns"]
        
        # Direct mapping attempts
        for csv_col in csv_columns:
            csv_lower = csv_col.lower().replace('_', '').replace(' ', '')
            
            for db_col in db_columns:
                db_lower = db_col.lower().replace('_', '')
                
                # Direct match
                if csv_lower == db_lower:
                    mapping_results["csv_to_db_mapping"][csv_col] = db_col
                    break
                # Partial match
                elif csv_lower in db_lower or db_lower in csv_lower:
                    if db_col not in mapping_results["mapping_suggestions"]:
                        mapping_results["mapping_suggestions"][db_col] = []
                    mapping_results["mapping_suggestions"][db_col].append(csv_col)
        
        # Find unmapped columns
        mapped_csv_cols = set(mapping_results["csv_to_db_mapping"].keys())
        mapping_results["unmapped_csv_columns"] = [
            col for col in csv_columns if col not in mapped_csv_cols
        ]
        
        # Find missing DB columns
        mapped_db_cols = set(mapping_results["csv_to_db_mapping"].values())
        mapping_results["missing_db_columns"] = [
            col for col in db_columns if col not in mapped_db_cols
        ]
    
    return mapping_results

def print_analysis_report(analysis: Dict[str, Any], mapping: Dict[str, Any]):
    """Print a formatted analysis report"""
    print("=" * 80)
    print("NFL DATA CSV ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nDirectory: {analysis['directory']}")
    print(f"Files analyzed: {analysis['files_analyzed']}")
    print(f"Files with errors: {analysis['files_with_errors']}")
    
    if "summary" in analysis:
        summary = analysis["summary"]
        print(f"\nDATA SUMMARY:")
        print(f"  Total unique columns: {summary['total_unique_columns']}")
        print(f"  Statistical columns: {summary['total_stat_columns']}")
        print(f"  Seasons found: {summary['seasons_found']}")
        print(f"  Positions found: {summary['positions_found']}")
        
        print(f"\nSTAT COLUMNS FOUND:")
        for col in summary['all_stat_columns'][:20]:  # Show first 20
            print(f"  - {col}")
        if len(summary['all_stat_columns']) > 20:
            print(f"  ... and {len(summary['all_stat_columns']) - 20} more")
    
    print(f"\nDATABASE MAPPING RESULTS:")
    print(f"  Direct mappings: {len(mapping['csv_to_db_mapping'])}")
    print(f"  Unmapped CSV columns: {len(mapping['unmapped_csv_columns'])}")
    print(f"  Missing DB columns: {len(mapping['missing_db_columns'])}")
    
    if mapping["csv_to_db_mapping"]:
        print(f"\nDIRECT MAPPINGS:")
        for csv_col, db_col in mapping["csv_to_db_mapping"].items():
            print(f"  {csv_col} → {db_col}")
    
    if mapping["mapping_suggestions"]:
        print(f"\nSUGGESTED MAPPINGS:")
        for db_col, csv_cols in mapping["mapping_suggestions"].items():
            print(f"  {db_col} ← {csv_cols}")
    
    if mapping["unmapped_csv_columns"]:
        print(f"\nUNMAPPED CSV COLUMNS:")
        unmapped_count = len(mapping["unmapped_csv_columns"])
        for col in mapping["unmapped_csv_columns"][:10]:  # Show first 10
            print(f"  - {col}")
        if unmapped_count > 10:
            print(f"  ... and {unmapped_count - 10} more")
    
    print("\n" + "=" * 80)

def main():
    """Main function - can handle single file or directory"""
    import sys
    
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default directory path
        target_path = "./data"
    
    if not os.path.exists(target_path):
        print(f"Path {target_path} not found!")
        return
    
    if os.path.isfile(target_path):
        # Single file mode
        print(f"Analyzing single CSV file: {target_path}")
        file_analysis = analyze_csv_file(target_path)
        
        # Create a mock directory analysis structure
        analysis = {
            "directory": os.path.dirname(target_path),
            "files_analyzed": 1,
            "files_with_errors": 1 if "error" in file_analysis else 0,
            "file_analyses": [file_analysis],
            "summary": {}
        }
        
        # Create summary for single file
        if "error" not in file_analysis:
            analysis["summary"] = {
                "total_unique_columns": len(file_analysis["columns"]),
                "total_stat_columns": len(file_analysis.get("stat_columns", [])),
                "all_columns": file_analysis["columns"],
                "all_stat_columns": file_analysis.get("stat_columns", []),
                "seasons_found": file_analysis["unique_values"].get("season", {}).get("values", []),
                "positions_found": file_analysis["unique_values"].get("position", {}).get("values", [])
            }
        
    else:
        # Directory mode
        print(f"Analyzing CSV files in directory: {target_path}")
        analysis = analyze_directory(target_path)
    
    # Map to database schema
    mapping = map_to_database_schema(analysis)
    
    # Print report
    print_analysis_report(analysis, mapping)
    
    # Save detailed results to JSON
    output_file = "nfl_csv_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "analysis": analysis,
            "mapping": mapping
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate summary statistics for study regions.

This script computes comprehensive statistics for each study region including:
- Number of 0.1x0.1 degree tiles
- Shot count statistics (total, mean/median/quartiles per tile)
- AGBD statistics
- Spatial extent
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from data.gedi import GEDIQuerier

# Study region definitions
REGIONS = {
    'maine': {
        'name': 'Maine, USA',
        'bbox': [-70, 44, -69, 45],
        'description': 'Temperate mixed forest, northeastern USA'
    },
    'sudtirol': {
        'name': 'South Tyrol, Italy',
        'bbox': [10.5, 45.6, 11.5, 46.4],
        'description': 'Alpine coniferous forest, European Alps'
    },
    'hokkaido': {
        'name': 'Hokkaido, Japan',
        'bbox': [143.8, 43.2, 144.8, 43.9],
        'description': 'Temperate deciduous/coniferous forest, northern Japan'
    },
    'tolima': {
        'name': 'Tolima, Colombia',
        'bbox': [-75, 3, -74, 4],
        'description': 'Tropical montane forest, Andean region'
    },
    'guaviare': {
        'name': 'Guaviare, Colombia',
        'bbox': [-73, 2, -72, 3],
        'description': 'Tropical rainforest, Amazon region'
    }
}


def compute_region_statistics(region_key: str, region_info: Dict[str, Any],
                              querier: GEDIQuerier) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a study region.

    Args:
        region_key: Region identifier (e.g., 'maine')
        region_info: Region metadata including bbox
        querier: GEDIQuerier instance

    Returns:
        Dictionary containing region statistics
    """
    print(f"\nProcessing {region_info['name']} ({region_key})...")

    # Query region data with tiles
    bbox = region_info['bbox']
    df = querier.query_region_tiles(
        region_bbox=tuple(bbox)
    )

    if len(df) == 0:
        print(f"  Warning: No data found for {region_key}")
        return None

    # Compute tile statistics
    n_tiles = df['tile_id'].nunique()
    shots_per_tile = df.groupby('tile_id').size()

    # Compute spatial extent
    lon_range = (df['longitude'].min(), df['longitude'].max())
    lat_range = (df['latitude'].min(), df['latitude'].max())
    area_deg2 = (lon_range[1] - lon_range[0]) * (lat_range[1] - lat_range[0])

    # Compute AGBD statistics
    agbd_stats = {
        'mean': float(df['agbd'].mean()),
        'median': float(df['agbd'].median()),
        'std': float(df['agbd'].std()),
        'min': float(df['agbd'].min()),
        'max': float(df['agbd'].max()),
        'q25': float(df['agbd'].quantile(0.25)),
        'q75': float(df['agbd'].quantile(0.75)),
    }

    # Compile statistics
    stats = {
        'region_key': region_key,
        'name': region_info['name'],
        'description': region_info.get('description', ''),
        'bbox': bbox,
        'spatial_extent': {
            'lon_range': lon_range,
            'lat_range': lat_range,
            'area_deg2': float(area_deg2),
        },
        'tiles': {
            'n_tiles': int(n_tiles),
            'expected_tiles': int((bbox[2] - bbox[0]) / 0.1) * int((bbox[3] - bbox[1]) / 0.1),
        },
        'shots': {
            'total': int(len(df)),
            'per_tile': {
                'mean': float(shots_per_tile.mean()),
                'median': float(shots_per_tile.median()),
                'std': float(shots_per_tile.std()),
                'min': int(shots_per_tile.min()),
                'max': int(shots_per_tile.max()),
                'q25': float(shots_per_tile.quantile(0.25)),
                'q50': float(shots_per_tile.quantile(0.50)),
                'q75': float(shots_per_tile.quantile(0.75)),
            }
        },
        'agbd': agbd_stats,
    }

    # Add tile coverage percentage
    if stats['tiles']['expected_tiles'] > 0:
        stats['tiles']['coverage_pct'] = 100 * stats['tiles']['n_tiles'] / stats['tiles']['expected_tiles']

    return stats


def print_summary_table(all_stats: Dict[str, Dict[str, Any]]):
    """Print a formatted summary table of all regions."""

    print("\n" + "="*120)
    print("STUDY REGION SUMMARY STATISTICS")
    print("="*120)

    # Create DataFrame for easy table formatting
    rows = []
    for region_key in ['maine', 'sudtirol', 'hokkaido', 'tolima', 'guaviare']:
        if region_key not in all_stats or all_stats[region_key] is None:
            continue

        stats = all_stats[region_key]
        rows.append({
            'Region': stats['name'],
            'Tiles': stats['tiles']['n_tiles'],
            'Coverage%': f"{stats['tiles'].get('coverage_pct', 0):.1f}",
            'Total Shots': f"{stats['shots']['total']:,}",
            'Shots/Tile Mean': f"{stats['shots']['per_tile']['mean']:.0f}",
            'Shots/Tile Med': f"{stats['shots']['per_tile']['median']:.0f}",
            'Shots/Tile Q25': f"{stats['shots']['per_tile']['q25']:.0f}",
            'Shots/Tile Q75': f"{stats['shots']['per_tile']['q75']:.0f}",
            'Shots/Tile Min': f"{stats['shots']['per_tile']['min']}",
            'Shots/Tile Max': f"{stats['shots']['per_tile']['max']}",
        })

    df_table = pd.DataFrame(rows)
    print("\n## Tile and Shot Statistics")
    print(df_table.to_string(index=False))

    # AGBD statistics table
    rows_agbd = []
    for region_key in ['maine', 'sudtirol', 'hokkaido', 'tolima', 'guaviare']:
        if region_key not in all_stats or all_stats[region_key] is None:
            continue

        stats = all_stats[region_key]
        agbd = stats['agbd']
        rows_agbd.append({
            'Region': stats['name'],
            'Mean': f"{agbd['mean']:.1f}",
            'Median': f"{agbd['median']:.1f}",
            'Std': f"{agbd['std']:.1f}",
            'Min': f"{agbd['min']:.1f}",
            'Q25': f"{agbd['q25']:.1f}",
            'Q75': f"{agbd['q75']:.1f}",
            'Max': f"{agbd['max']:.1f}",
        })

    df_agbd = pd.DataFrame(rows_agbd)
    print("\n## AGBD Statistics (Mg/ha)")
    print(df_agbd.to_string(index=False))

    # Spatial extent table
    rows_spatial = []
    for region_key in ['maine', 'sudtirol', 'hokkaido', 'tolima', 'guaviare']:
        if region_key not in all_stats or all_stats[region_key] is None:
            continue

        stats = all_stats[region_key]
        extent = stats['spatial_extent']
        rows_spatial.append({
            'Region': stats['name'],
            'Lon Range': f"[{extent['lon_range'][0]:.2f}, {extent['lon_range'][1]:.2f}]",
            'Lat Range': f"[{extent['lat_range'][0]:.2f}, {extent['lat_range'][1]:.2f}]",
            'Area (deg²)': f"{extent['area_deg2']:.2f}",
        })

    df_spatial = pd.DataFrame(rows_spatial)
    print("\n## Spatial Extent")
    print(df_spatial.to_string(index=False))
    print("\n" + "="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary statistics for study regions'
    )
    parser.add_argument(
        '--regions',
        nargs='+',
        choices=list(REGIONS.keys()) + ['all'],
        default=['all'],
        help='Regions to process (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='region_summary_stats.json',
        help='Output JSON file path (default: region_summary_stats.json)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable query caching'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for GEDI queries (optional)'
    )

    args = parser.parse_args()

    # Determine which regions to process
    if 'all' in args.regions:
        regions_to_process = list(REGIONS.keys())
    else:
        regions_to_process = args.regions

    # Handle caching - if --no-cache is set, disable cache_dir
    cache_dir = None if args.no_cache else args.cache_dir

    # Initialize querier
    print(f"Initializing GEDI querier...")
    if cache_dir:
        print(f"  Cache directory: {cache_dir}")
    else:
        print(f"  Caching disabled")
    querier = GEDIQuerier(cache_dir=cache_dir)

    # Compute statistics for each region
    all_stats = {}
    for region_key in regions_to_process:
        if region_key not in REGIONS:
            print(f"Warning: Unknown region '{region_key}', skipping...")
            continue

        region_info = REGIONS[region_key]
        stats = compute_region_statistics(
            region_key,
            region_info,
            querier
        )

        if stats is not None:
            all_stats[region_key] = stats

    # Print summary table
    if all_stats:
        print_summary_table(all_stats)

        # Save to JSON
        output_path = Path(args.output)
        print(f"Saving statistics to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(all_stats, f, indent=2)

        print(f"✓ Summary statistics saved to {output_path}")
    else:
        print("No statistics computed.")


if __name__ == '__main__':
    main()

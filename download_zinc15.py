"""
Download ZINC15 molecules from .uri file (list of URLs to .smi tranches).
Downloads only enough tranches to get the target number of molecules.

Usage:
    python download_zinc15.py --uri ZINC-downloader-2D-smi.uri --output data/zinc15_druglike.smi --target 50000
"""
import argparse
import urllib.request
import os
import random


def download_zinc(uri_file, output_path, target_n=50000, max_tranches=100, seed=42):
    """Download ZINC15 .smi tranches until we have enough molecules."""
    
    # Read all URLs
    with open(uri_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip().startswith('http')]
    
    print(f"Found {len(urls)} ZINC15 tranches in {uri_file}")
    print(f"Target: {target_n} molecules (will stop early once reached)")
    print(f"Max tranches to download: {max_tranches}")
    print()
    
    # Shuffle so we get diverse tranches (not just the first MW range)
    random.seed(seed)
    random.shuffle(urls)
    urls = urls[:max_tranches]
    
    all_smiles = []
    
    for i, url in enumerate(urls):
        if len(all_smiles) >= target_n:
            print(f"\n  ✓ Reached target of {target_n} molecules after {i} tranches")
            break
        
        tranche_name = url.split('/')[-1]
        try:
            print(f"  [{i+1}/{len(urls)}] Downloading {tranche_name}...", end=' ', flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (ZINC15 download)'})
            response = urllib.request.urlopen(req, timeout=30)
            content = response.read().decode('utf-8')
            
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            smiles = [l.split()[0] for l in lines if l and not l.startswith('#')]
            
            all_smiles.extend(smiles)
            print(f"{len(smiles)} molecules (total: {len(all_smiles)})")
            
        except Exception as e:
            print(f"FAILED ({e})")
            continue
    
    # Deduplicate
    unique_smiles = list(set(all_smiles))
    print(f"\nTotal: {len(all_smiles)} → {len(unique_smiles)} unique molecules")
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for smi in unique_smiles:
            f.write(f"{smi}\n")
    
    print(f"✓ Saved to {output_path}")
    return unique_smiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ZINC15 molecules from .uri file')
    parser.add_argument('--uri', required=True, help='Path to ZINC .uri file')
    parser.add_argument('--output', default='data/zinc15_druglike.smi', help='Output .smi file')
    parser.add_argument('--target', type=int, default=50000, help='Target number of molecules')
    parser.add_argument('--max_tranches', type=int, default=100, help='Max tranches to download')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for tranche selection')
    args = parser.parse_args()
    
    download_zinc(args.uri, args.output, args.target, args.max_tranches, args.seed)

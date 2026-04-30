# =============================================================
# Neglect-Fold | Automated PDB Structure Search
# =============================================================
# Searches PDB database systematically for:
# 1. NTD parasite proteins with bound ligands
# 2. Human proteins with bound ligands
#
# Filters by:
# - Has drug-like ligand (not just water/ions)
# - Good resolution (<2.5 Angstroms)
# - Reasonable size (100-1000 residues)
#
# Goal: find 1000+ quality training structures

import requests
import json
import pandas as pd
import os
import time

OUTPUT_DIR = "data/processed/pdb_search"

# ============================================================
# PDB organism taxonomy IDs
# ============================================================
ORGANISM_IDS = {
    "trypanosoma_cruzi": "353153",
    "leishmania_donovani": "5661",
    "leishmania_major": "347515",
    "schistosoma_mansoni": "6183",
    "homo_sapiens": "9606"  # Human
}

def search_pdb_by_organism(tax_id, organism_name, max_results=500):
    """
    Searches PDB for all structures from a specific organism
    that have bound ligands.
    
    Uses PDB's REST API with JSON query.
    """
    print(f"\nSearching PDB for {organism_name}...")
    
    # PDB search query
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                        "operator": "exact_match",
                        "value": tax_id
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ]
        }
    }
    
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    try:
        response = requests.post(
            url,
            json=query,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code}")
            return []
        
        data = response.json()
        results = data.get('result_set', [])
        total = data.get('total_count', 0)
        
        pdb_ids = [r['identifier'] for r in results]
        
        print(f"  Found {total} total structures")
        print(f"  Retrieved {len(pdb_ids)} PDB IDs")
        
        return pdb_ids
        
    except Exception as e:
        print(f"  Search error: {e}")
        return []

def get_structure_details(pdb_id):
    """
    Gets details about a PDB structure to check quality.
    Returns resolution, chain count, ligand info.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Get resolution
        resolution = data.get(
            'rcsb_entry_info', {}
        ).get('resolution_combined', [None])
        if isinstance(resolution, list):
            resolution = resolution[0] if resolution else None
        
        # Get polymer count (number of protein chains)
        polymer_count = data.get(
            'rcsb_entry_info', {}
        ).get('polymer_entity_count', 0)
        
        # Get non-polymer count (ligands)
        nonpolymer_count = data.get(
            'rcsb_entry_info', {}
        ).get('nonpolymer_entity_count', 0)
        
        return {
            'pdb_id': pdb_id,
            'resolution': resolution,
            'polymer_count': polymer_count,
            'nonpolymer_count': nonpolymer_count
        }
        
    except Exception as e:
        return None

def filter_structures(pdb_ids, max_to_check=200):
    """
    Filters PDB structures by quality criteria.
    
    Keeps structures that:
    - Have resolution < 2.5 Angstroms
    - Have at least 1 ligand
    - Have at most 4 protein chains
    """
    print(f"  Filtering {min(len(pdb_ids), max_to_check)} structures...")
    
    good_structures = []
    checked = 0
    
    for pdb_id in pdb_ids[:max_to_check]:
        details = get_structure_details(pdb_id)
        
        if details is None:
            continue
        
        # Apply filters
        resolution = details['resolution']
        if resolution and resolution > 2.5:
            continue
        
        if details['nonpolymer_count'] == 0:
            continue
            
        if details['polymer_count'] > 4:
            continue
        
        good_structures.append(pdb_id)
        checked += 1
        
        if checked % 20 == 0:
            print(f"  Checked {checked}, kept {len(good_structures)}")
        
        time.sleep(0.1)
    
    print(f"  Quality structures: {len(good_structures)}")
    return good_structures

# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: PDB Structure Search ===")
    print("Finding high quality training structures")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    # Search for each organism
    for organism_name, tax_id in ORGANISM_IDS.items():
        # Search PDB
        pdb_ids = search_pdb_by_organism(
            tax_id, 
            organism_name,
            max_results=2000
        )
        
        if not pdb_ids:
            continue
        
        # Filter by quality
        good_ids = filter_structures(pdb_ids, max_to_check=1000)
        
        # Determine label
        label = 0 if organism_name == "homo_sapiens" else 1
        
        for pdb_id in good_ids:
            all_results.append({
                'pdb_id': pdb_id,
                'organism': organism_name,
                'label': label
            })
        
        print(f"  Final count for {organism_name}: {len(good_ids)}")
        time.sleep(1)
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(f"{OUTPUT_DIR}/pdb_search_results.csv", index=False)
    
    # Summary
    print(f"\n{'='*50}")
    print("SEARCH SUMMARY")
    print(f"{'='*50}")
    if len(df) > 0:
        print(df.head(10))
        print(f"Columns: {list(df.columns)}")
    else:
        print("No results found!")
    print(f"\nTotal structures found: {len(df)}")
    print(f"Parasite: {len(df[df['label']==1])}")
    print(f"Human: {len(df[df['label']==0])}")
    print(f"\nSaved to {OUTPUT_DIR}/pdb_search_results.csv")
    print("\n=== Search Complete! ===")
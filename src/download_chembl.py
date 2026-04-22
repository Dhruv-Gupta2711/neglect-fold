# =============================================================
# Neglect-Fold | Phase 1: Download Known Inhibitors from ChEMBL
# =============================================================

import requests
import pandas as pd
import numpy as np
import os
import time

# ---- ChEMBL taxonomy IDs for our organisms ----
ORGANISMS = {
    "trypanosoma_cruzi": "Trypanosoma cruzi",
    "leishmania_donovani": "Leishmania donovani",
    "schistosoma_mansoni": "Schistosoma mansoni"
}

OUTPUT_DIR = "data/raw"

def download_bioactivities(organism_name, organism_full_name):
    """
    Downloads bioactivity data from ChEMBL for one organism.
    """
    print(f"\nDownloading ChEMBL data for {organism_name}...")
    
    all_results = []
    offset = 0
    limit = 1000
    
    while True:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            f"target_organism={organism_full_name}&"
            f"standard_type__in=IC50,EC50,Ki&"
            f"standard_value__isnull=false&"
            f"canonical_smiles__isnull=false&"
            f"limit={limit}&"
            f"offset={offset}"
        )
        
        response = requests.get(url, timeout=30)
        print(f"  Status code: {response.status_code}, offset: {offset}")
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code}")
            break
        
        data = response.json()
        activities = data.get('activities', [])
        
        if not activities:
            print(f"  No more results at offset {offset}")
            break
        
        for activity in activities:
            all_results.append({
                "molecule_chembl_id": activity.get("molecule_chembl_id"),
                "target_chembl_id": activity.get("target_chembl_id"),
                "standard_type": activity.get("standard_type"),
                "standard_value": activity.get("standard_value"),
                "standard_units": activity.get("standard_units"),
                "smiles": activity.get("canonical_smiles"),
                "organism": organism_name
            })
        
        total_count = data.get('page_meta', {}).get('total_count', 0)
        print(f"  Got {len(activities)} records, total so far: {len(all_results)} of {total_count}")
        
        offset += limit
        if offset >= total_count:
            break
            
        time.sleep(0.5)
    
    return all_results

def clean_bioactivities(df, organism_name):
    """
    Cleans raw ChEMBL data and converts to pIC50.
    """
    print(f"\nCleaning {organism_name} data...")
    print(f"Starting records: {len(df)}")
    
    df = df.dropna(subset=['smiles', 'standard_value'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    df = df[df['standard_value'] > 0]
    
    # Convert to pIC50
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    df = df[(df['pIC50'] >= 2) & (df['pIC50'] <= 15)]
    
    print(f"Final clean records: {len(df)}")
    return df

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: ChEMBL Downloader ===")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    all_dataframes = []
    
    for organism_name, organism_full_name in ORGANISMS.items():
        results = download_bioactivities(organism_name, organism_full_name)
        
        if not results:
            print(f"No data found for {organism_name}")
            continue
        
        df = pd.DataFrame(results)
        
        # Save raw
        raw_path = f"{OUTPUT_DIR}/{organism_name}_chembl_raw.csv"
        df.to_csv(raw_path, index=False)
        
        # Clean and save
        df_clean = clean_bioactivities(df, organism_name)
        clean_path = f"data/processed/{organism_name}_chembl_clean.csv"
        df_clean.to_csv(clean_path, index=False)
        
        all_dataframes.append(df_clean)
        print(f"Final {organism_name}: {len(df_clean)} compounds")
    
    if all_dataframes:
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df.to_csv("data/processed/all_chembl_clean.csv", index=False)
        print(f"\nMaster dataset: {len(master_df)} total compounds")
    
    print("\n=== ChEMBL Download Complete! ===")
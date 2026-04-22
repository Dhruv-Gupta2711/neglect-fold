# =============================================================
# Neglect-Fold | Phase 1: Download Proteomes from UniProt
# =============================================================
# UniProt is the world's largest protein database
# We will download all proteins for our 3 target organisms

import requests
import os
import time

# ---- Define our three target organisms ----
# These are the official UniProt taxonomy IDs
# Think of these like a unique ID number for each organism

ORGANISMS = {
    "trypanosoma_cruzi": "353153",    # Causes Chagas disease
    "leishmania_donovani": "5661",    # Causes Leishmaniasis  
    "schistosoma_mansoni": "6183"     # Causes Schistosomiasis
}

# ---- Where to save the data ----
OUTPUT_DIR = "data/raw"

def download_proteome(organism_name, tax_id):
    """
    Downloads all proteins for one organism from UniProt.
    
    organism_name: what we call the organism (e.g. "trypanosoma_cruzi")
    tax_id: UniProt's ID number for this organism
    """
    
    print(f"\nDownloading proteins for {organism_name}...")
    
    # Build the URL to query UniProt's API
    # This is like a search query sent to UniProt's website
    url = (
        f"https://rest.uniprot.org/uniprotkb/stream?"
        f"query=organism_id:{tax_id}&"
        f"format=fasta&"
        f"compressed=false"
    )
    
    print(f"Querying UniProt API...")
    
    # Send the request to UniProt
    response = requests.get(url, stream=True)
    
    # Check if it worked
    if response.status_code != 200:
        print(f"ERROR: Could not download {organism_name}")
        return
    
    # Save the data to a file
    output_file = os.path.join(OUTPUT_DIR, f"{organism_name}.fasta")
    
    with open(output_file, 'w') as f:
        f.write(response.text)
    
    # Count how many proteins we got
    protein_count = response.text.count('>')
    print(f"Saved {protein_count} proteins to {output_file}")

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Proteome Downloader ===")
    
    # Make sure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download proteins for each organism
    for organism_name, tax_id in ORGANISMS.items():
        download_proteome(organism_name, tax_id)
        time.sleep(1)  # Be polite to UniProt's servers
    
    print("\nAll downloads complete!")
    print(f"Check your {OUTPUT_DIR} folder for the files.")
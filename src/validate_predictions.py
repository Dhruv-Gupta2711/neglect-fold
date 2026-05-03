# =============================================================
# Neglect-Fold | Literature Validation
# =============================================================
import requests
import pandas as pd
import time
import os

def get_uniprot_info(protein_id):
    clean_id = protein_id.split('|')[1] if '|' in protein_id else protein_id
    url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        info = {
            'protein_id': clean_id,
            'protein_name': '',
            'function': '',
            'disease': '',
            'drug_target': False,
        }
        
        # Protein name
        try:
            info['protein_name'] = (
                data['proteinDescription']
                ['recommendedName']
                ['fullName']['value']
            )
        except:
            try:
                info['protein_name'] = (
                    data['proteinDescription']
                    ['submissionNames'][0]
                    ['fullName']['value']
                )
            except:
                pass
        
        # Function
        try:
            for comment in data.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        info['function'] = texts[0].get(
                            'value', ''
                        )[:300]
                        break
        except:
            pass
        
        # Check if known drug target
        try:
            db_refs = data.get('dbReferences', [])
            drug_dbs = ['DrugBank', 'ChEMBL', 'BindingDB']
            for ref in db_refs:
                if ref.get('database') in drug_dbs:
                    info['drug_target'] = True
                    break
        except:
            pass
        
        return info
        
    except Exception as e:
        return None

def classify_target(info):
    if info is None:
        return "UNKNOWN"
    if info['drug_target']:
        return "KNOWN TARGET"
    if info['function']:
        return "HYPOTHESIZED"
    return "NOVEL CANDIDATE"

if __name__ == "__main__":
    print("=== Neglect-Fold: Literature Validation ===")
    
    results_path = "results/top20_drug_targets.csv"
    if not os.path.exists(results_path):
        print("Run pipeline.py first!")
        exit()
    
    df = pd.read_csv(results_path)
    print(f"Validating {len(df)} predictions\n")
    
    validation_results = []
    
    for _, row in df.iterrows():
        protein_id = row['protein_id']
        organism = row['organism']
        rank = row['rank']
        
        print(f"Rank {rank}: {protein_id} ({organism})")
        
        info = get_uniprot_info(protein_id)
        
        if info:
            classification = classify_target(info)
            print(f"  Name: {info['protein_name']}")
            print(f"  Function: {info['function'][:100]}")
            print(f"  Classification: {classification}")
            print(f"  Known drug target: {info['drug_target']}")
        else:
            classification = "UNKNOWN"
            info = {
                'protein_name': 'Unknown',
                'function': 'Unknown',
                'drug_target': False
            }
        
        validation_results.append({
            'rank': rank,
            'protein_id': protein_id,
            'organism': organism,
            'final_score': row['final_score'],
            'protein_name': info['protein_name'],
            'function': info['function'][:200],
            'classification': classification,
            'known_drug_target': info['drug_target']
        })
        
        time.sleep(0.5)
    
    val_df = pd.DataFrame(validation_results)
    val_df.to_csv(
        'results/validation_results.csv',
        index=False
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    known = len(val_df[val_df['classification']=='KNOWN TARGET'])
    hypothesized = len(val_df[val_df['classification']=='HYPOTHESIZED'])
    novel = len(val_df[val_df['classification']=='NOVEL CANDIDATE'])
    unknown = len(val_df[val_df['classification']=='UNKNOWN'])
    
    print(f"Known drug targets:     {known}")
    print(f"Hypothesized targets:   {hypothesized}")
    print(f"Novel candidates:       {novel}")
    print(f"Unknown:                {unknown}")
    print(f"\nSaved to results/validation_results.csv")
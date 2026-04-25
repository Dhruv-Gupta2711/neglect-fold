import pandas as pd
df = pd.read_csv('results/top20_drug_targets.csv')
print(df[['pocket_score','human_similarity_pct','best_compound_pIC50']].describe())
print("\nActual values:")
print(df[['protein_id','pocket_score','human_similarity_pct','best_compound_pIC50']].to_string())
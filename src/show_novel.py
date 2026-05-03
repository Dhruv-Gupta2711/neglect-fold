import pandas as pd

df = pd.read_csv('results/validation_results.csv')

print("=== NOVEL DRUG TARGET CANDIDATES ===")
print("These proteins have never been suggested as drug targets!")
print()

novel = df[df['classification'] == 'NOVEL CANDIDATE']
for _, row in novel.iterrows():
    print(f"Rank {row['rank']}: {row['protein_id']}")
    print(f"  Disease: {row['organism'].replace('_', ' ').title()}")
    print(f"  Score: {row['final_score']}")
    print(f"  Name: {row['protein_name']}")
    print()

print("=== HYPOTHESIZED TARGETS ===")
print("These are in literature but never successfully drugged")
print()

hyp = df[df['classification'] == 'HYPOTHESIZED']
for _, row in hyp.iterrows():
    print(f"Rank {row['rank']}: {row['protein_id']}")
    print(f"  Disease: {row['organism'].replace('_', ' ').title()}")
    print(f"  Score: {row['final_score']}")
    print(f"  Name: {row['protein_name']}")
    print()
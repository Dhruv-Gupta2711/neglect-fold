import pandas as pd
df = pd.read_csv('results/validation_results.csv')
print(df[['rank','organism','protein_id','classification','protein_name']].to_string())
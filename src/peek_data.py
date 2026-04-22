# Quick peek at the downloaded protein data

with open('data/raw/trypanosoma_cruzi.fasta') as f:
    lines = f.readlines()

print("First 10 lines of Trypanosoma cruzi proteins:")
print("=" * 50)
for line in lines[:10]:
    print(line.strip())
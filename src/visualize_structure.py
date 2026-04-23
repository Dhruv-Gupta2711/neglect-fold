# =============================================================
# Neglect-Fold | Phase 2: Visualize a Protein Structure
# =============================================================
# We'll create an HTML file that shows the 3D protein structure
# interactively in your browser

import os

# Read the PDB file
structure_dir = "data/processed/structures/trypanosoma_cruzi"
pdb_files = os.listdir(structure_dir)
first_file = os.path.join(structure_dir, pdb_files[0])

with open(first_file) as f:
    pdb_data = f.read()

protein_name = pdb_files[0].replace('.pdb', '')

# Create an interactive HTML visualization
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neglect-Fold: {protein_name} Structure</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ 
            margin: 0; 
            background: #1a1a2e;
            font-family: Arial, sans-serif;
            color: white;
        }}
        #header {{
            padding: 20px;
            text-align: center;
            background: #16213e;
        }}
        h1 {{ color: #00d4ff; margin: 0; }}
        p {{ color: #aaa; margin: 5px 0; }}
        #viewer {{ 
            width: 100%; 
            height: 80vh; 
            position: relative;
        }}
        #info {{
            padding: 15px;
            text-align: center;
            background: #16213e;
            font-size: 14px;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🧬 Neglect-Fold Structure Viewer</h1>
        <p>Protein: <strong style="color:#00d4ff">{protein_name}</strong> 
        | Organism: <strong style="color:#00ff88">Trypanosoma cruzi</strong> 
        | Source: AlphaFold DB</p>
        <p>This is the predicted 3D shape of a real Chagas disease parasite protein</p>
    </div>
    
    <div id="viewer"></div>
    
    <div id="info">
        🖱️ Left click + drag to rotate | Scroll to zoom | Right click + drag to translate
        | Colors show AlphaFold confidence (Blue = high, Red = low)
    </div>

    <script>
        const pdbData = `{pdb_data}`;
        
        let viewer = $3Dmol.createViewer(
            document.getElementById('viewer'),
            {{ backgroundColor: '0x1a1a2e' }}
        );
        
        viewer.addModel(pdbData, 'pdb');
        
        // Color by AlphaFold confidence (pLDDT score)
        // Blue = very confident, Red = less confident
        viewer.setStyle({{}}, {{
            cartoon: {{
                colorfunc: function(atom) {{
                    const b = atom.b; // pLDDT score stored in B-factor
                    if (b >= 90) return '#0053D6';      // Very high - dark blue
                    if (b >= 70) return '#65CBF3';      // High - light blue  
                    if (b >= 50) return '#FFDB13';      // Medium - yellow
                    return '#FF7D45';                    // Low - orange
                }}
            }}
        }});
        
        viewer.zoomTo();
        viewer.render();
        viewer.zoom(0.8);
    </script>
</body>
</html>
"""

# Save the HTML file
output_path = "results/figures/protein_viewer.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"3D viewer created!")
print(f"Opening: {output_path}")
print(f"Protein: {protein_name}")

# Open in browser automatically
import webbrowser
import pathlib
webbrowser.open(pathlib.Path(output_path).resolve().as_uri())
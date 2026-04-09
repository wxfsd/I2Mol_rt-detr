import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm


# Directory containing SDF files
sdf_dir = '/home/jovyan/rt-detr/data/OCSR_Review/assets/reference/JPO_mol_ref/'

# Create an empty list to store the data
data = []

# Loop through all SDF files in the directory
for sdf_file in tqdm(os.listdir(sdf_dir)):
    if sdf_file.endswith('.sdf'):
        sdf_path = os.path.join(sdf_dir, sdf_file)
        
        # Read the SDF file
        suppl = Chem.SDMolSupplier(sdf_path)
        
        # Loop through each molecule in the SDF file
        for mol in suppl:
            if mol is not None:
                # Kekulize the molecule (convert to kekulized form)
                Chem.Kekulize(mol, clearAromaticFlags=True)
                # Get the SMILES representation of the molecule
                smiles = Chem.MolToSmiles(mol)
                # Append the data (SMILES and file name)
                sdf2png_=sdf_file.split('.')[0] + '.png'
                data.append([smiles, sdf2png_])

# Convert the list of data into a pandas DataFrame
df = pd.DataFrame(data, columns=['SMILES', 'file_name'])

# Save the DataFrame to a CSV file
df.to_csv('/home/jovyan/rt-detr/data/real_processed/JPO_with_charge/JPO.csv', index=False)

print("CSV file saved successfully.")

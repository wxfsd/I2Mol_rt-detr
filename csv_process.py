# import pandas as pd

# df = pd.read_excel('/home/jovyan/data/OCR_SMILES/smiles_atom_32-x.xlsx',header=0)

# df['file_name'] = df['file_name'].apply(lambda x: str(x+20000) + '.png')

# df.to_csv('/home/jovyan/data/OCR_SMILES/smiles_atom_32-x.csv', index=False)


import pandas as pd

df1 = pd.read_csv('/home/jovyan/data/OCR_SMILES/smiles_atom_x-18.csv')
df2 = pd.read_csv('/home/jovyan/data/OCR_SMILES/smiles_atom_24-26.csv')
df3 = pd.read_csv('/home/jovyan/data/OCR_SMILES/smiles_atom_32-x.csv')

merged_df = pd.concat([df1, df2, df3], axis=0)

merged_df.to_csv('/home/jovyan/data/OCR_SMILES/merged_file.csv', index=False)


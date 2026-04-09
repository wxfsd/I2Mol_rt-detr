from rdkit_AugSmiles import generate_image
import cv2
import rdkit
from rdkit import Chem

idx = 1
smiles = 'Br[O+2]1(C2=CC=CC=C2)[N-]C1([FH+])[I-]C'
img, smiles, graph, success = generate_image(idx, smiles)
print(smiles)  #[H][F+]C1([O+2](c2cc([C@@]3([C@@]4(C[C@](C=C4)([H])[C@]3(CC(=O)N)[H])[H])C(O)=O)ccc2)(Br)[N-]1)[I-]C
cv2.imwrite("img.png",img)
mol = Chem.MolFromSmiles(smiles)
Chem.Draw.MolToImageFile(mol, 'image.png')


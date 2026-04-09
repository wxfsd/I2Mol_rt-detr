import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from PIL import Image, ImageDraw


rdkitbond_type_dict = {
            rdkit.Chem.rdchem.BondType.SINGLE:1,
            rdkit.Chem.rdchem.BondType.DOUBLE:2,
            rdkit.Chem.rdchem.BondType.TRIPLE:3,
            rdkit.Chem.rdchem.BondType.AROMATIC:4,
            }
rdkitbond_dir_dict = {
            rdkit.Chem.rdchem.BondDir.BEGINWEDGE:5,
            rdkit.Chem.rdchem.BondDir.BEGINDASH:6,
            rdkit.Chem.rdchem.BondDir.UNKNOWN:7,
            rdkit.Chem.rdchem.BondDir.ENDUPRIGHT:8,
            rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT:9,
}

i2rdkitBond={
   1:rdkit.Chem.rdchem.BondType.SINGLE,
   2:rdkit.Chem.rdchem.BondType.DOUBLE,
   3:rdkit.Chem.rdchem.BondType.TRIPLE,
   4:rdkit.Chem.rdchem.BondType.AROMATIC,
   5:rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
   6:rdkit.Chem.rdchem.BondDir.BEGINDASH,
   7:rdkit.Chem.rdchem.BondDir.UNKNOWN,
   8:rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
   9:rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
   }

# 原smiles   
# ori="O=C([C@@H]1[C@H](/C=C/C2=CC=CC=C2)C[C@@H]1/C=C/C3=CC=CC=C3)NC4=C5C(C=CC=N5)=CC=C4"
ori = 'COC1C=NC(=CC=1)[C@@H]1C[C@H]1COC1=NC(C)=NC=C1C1=CN=C(C)N1C'
# 重构的smiles  
# O=C(Nc1cccc2cccnc12)[C@H]1C(C=Cc2ccccc2)C[C@H]1C=Cc1ccccc1
# m = Chem.MolFromSmiles('CC123.F2.Cl1.Br3')
m = Chem.MolFromSmiles(ori)
orican=Chem.MolToSmiles(m)
Chem.rdDepictor.Compute2DCoords(m)
print(m.GetNumConformers())
Chem.rdmolops.WedgeMolBonds(m,m.GetConformer(0))
print(Chem.MolToSmiles(m))


#atom bonds charge corrds
symbols=[]
charges=[]
coords=[]
atoms = [atom for atom in m.GetAtoms()]#TODO get atom net charge for using
for i, atom in enumerate(atoms):
    positions = m.GetConformer().GetAtomPosition(i)
    symbols.append(atom.GetSymbol())
    charges.append(atom.GetFormalCharge())       
    print(atom.GetSymbol(), positions.x, positions.y, positions.z)
    coords.append([positions.x, positions.y,0])

bonds_list=[]
for jj,bond in enumerate(m.GetBonds()):
    s = bond.GetBeginAtomIdx()#rdkit not random atom order now
    t = bond.GetEndAtomIdx()
    if s in [2,3,13] or t in [2,3,13]:
        print(s,t,bond.GetBondDir())
    #not considering double bond dir
    if  bond.GetBondDir() in [rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
                            rdkit.Chem.rdchem.BondDir.BEGINDASH,
                            rdkit.Chem.rdchem.BondDir.UNKNOWN,
                            rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
                            rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,]:#NOTE indigo now no wavy bond only rdkit
        bond_type=rdkitbond_dir_dict[bond.GetBondDir()]
    else:
        bond_type=rdkitbond_type_dict[bond.GetBondType()]
        
    # print(bond.GetBondDir(),bond_type,[s,t])
    bonds_list.append((bond_type,s,t))

# me = Chem.EditableMol(Chem.Mol())
me = Chem.RWMol()#read and write, better than the above

#rebuilt
for a,c in zip(symbols,charges):
    # print(a,c)
    if a in  ["R",'*']:     
        new_atom = Chem.Atom("*")
        new_atom.SetProp("atomLabel", "abb") 
        # mol.AddAtom(new_atom)
        me.AddAtom(new_atom)
    atom = Chem.Atom(a)#TODO when Rgroup rdkit not reconized
    atom.SetFormalCharge(c)
    me.AddAtom(atom)
for bond_type,s,t in bonds_list:
    if bond_type<=4:
        me.AddBond(s, t, i2rdkitBond[bond_type])
    else:
        me.AddBond(s, t, rdkit.Chem.rdchem.BondType.SINGLE)
#atom chiralty
mol_tmp = me.GetMol()
chiral_centers = Chem.FindMolChiralCenters(mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
chiral_center_ids = [idx for idx, _ in chiral_centers] 
print(chiral_centers,chiral_center_ids)
chiral_center_ids_neibor={}
for chi in chiral_center_ids:
    ats=mol_tmp.GetAtomWithIdx(chi).GetNeighbors()
    chiral_center_ids_neibor[chi]=[at.GetIdx() for at in ats]
    print(chiral_center_ids_neibor)

me.Debug()
opts = Draw.MolDrawOptions()
opts.addAtomIndices = True
opts.addStereoAnnotation = True
img = Draw.MolToImage(me, options=opts)
img.save('/home/jovyan/rt-detr/output/test/0.png')

#set 3d get E/Z bond dir
me.RemoveAllConformers()
conf = Chem.Conformer(me.GetNumAtoms())
conf.Set3D(True)
for i, (x, y,z) in enumerate(coords):
    conf.SetAtomPosition(i, (x, y, z))
me.AddConformer(conf)
Chem.SanitizeMol(me)
Chem.AssignStereochemistryFrom3D(me)
# NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
# So we do this first, remove the conformer and add back the 2D conformer for chiral correction

me.Debug()
opts = Draw.MolDrawOptions()
opts.addAtomIndices = True
opts.addStereoAnnotation = True
img = Draw.MolToImage(me, options=opts)
img.save('/home/jovyan/rt-detr/output/test/1.png')

#add bond direaction NOTE do not delete bond as the EZ may missing
for i in chiral_center_ids:
    for j in chiral_center_ids_neibor[i]:
        # print(i,j)
        # if edges[i][j] == 5:
        if (5,i,j) in bonds_list  :
            # me.RemoveBond(i, j)
            # me.AddBond(i, j, Chem.BondType.SINGLE)
            # me.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            me.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
            # print([i,j],5)
        elif (6,i,j) in bonds_list  :
            # me.RemoveBond(i, j)
            # me.AddBond(i, j, Chem.BondType.SINGLE)
            me.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            # print([i,j],6)

me.Debug()
opts = Draw.MolDrawOptions()
opts.addAtomIndices = True
opts.addStereoAnnotation = True
img = Draw.MolToImage(me, options=opts)
img.save('/home/jovyan/rt-detr/output/test/2.png')


# magic
Chem.SanitizeMol(me)
Chem.DetectBondStereochemistry(me)
Chem.AssignChiralTypesFromBondDirs(me)
Chem.AssignStereochemistry(me)

opts = Draw.MolDrawOptions()
opts.addAtomIndices = True
opts.addStereoAnnotation = True
img = Draw.MolToImage(me, options=opts)
img.save('/home/jovyan/rt-detr/output/test/3.png')

rebuilt_sm=Chem.MolToSmiles(me)
print(f'{rebuilt_sm}\n{orican}')
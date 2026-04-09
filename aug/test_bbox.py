import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from xml.dom import minidom
import os
import re
import numpy as np

def _get_svg_doc(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    doc = minidom.parseString(svg)
    return doc,svg


def extract_charge(atom_str):
    charge_match = re.search(r'(\+|-)(\d+)?', atom_str)
    if charge_match:
        sign = charge_match.group(1)
        if charge_match.group(2):
            charge = int(charge_match.group(2)) 
        else:
            charge = 1 
        return charge if sign == '+' else -charge
    else:
        return 0 


# smiles="Br[O+2]1(C2=CC=CC=C2)[N-]C1([FH+])[I-]C"
smiles_list = ['[O-][S+2]([O-])(NCCC(=O)NCC1C=CC=C(Cl)C=1)C1C=CC(Br)=CC=1','Br[O+2]1(C2=CC=CC=C2)[N-]C1([FH+])[I-]C']
# smiles_list = ['[O-][S+2]([O-])(NCCC(=O)NCC1C=CC=C(Cl)C=1)C1C=CC(Br)=CC=1']


for smiles in smiles_list:
    mol = rdkit.Chem.MolFromSmiles(smiles)
    atom_count = mol.GetNumAtoms()

    opts = Draw.MolDrawOptions()
    opts.addAtomIndices = True
    opts.addStereoAnnotation = True
    Draw.MolToImage(mol, kekulize=False,options=opts)
    Chem.Draw.MolToImageFile(mol, '/home/jovyan/rt-detr/output/test.png')

    doc,svg=_get_svg_doc(mol)
    annotations = []
    atom_with_charge = {}
    charges_ats=[]
    charges_ids=[]
    for path in doc.getElementsByTagName('rdkit:atom'):
        a_str=path.getAttribute('atom-smiles')
        a_id=path.getAttribute('idx')
        # #get charge coords
        if '-' in a_str or  '+' in a_str:
            charges_ats.append((a_str,int(a_id)-1))#need zero start
            charges_ids.append(f"atom-{int(a_id)-1}")
            print([path.getAttribute('atom-smiles'),path.getAttribute('idx')])
            atom_with_charge[f'atom-{int(a_id)-1}'] = path.getAttribute('atom-smiles')

    
    if atom_count >= 18:
        radius = 2
    else:
        radius  = 5
    # doc
    str_list = []
    average_coords_list = []
    pre = None
    for path in doc.getElementsByTagName('path'):
        str_xxx=path.getAttribute('class')
        if 'bond' not in str_xxx and 'atom' in str_xxx :
            if str_xxx in charges_ids:
                coords_str=path.getAttribute('d')
                # print([str_xxx],coords)
                # Extract the numbers using regular expressions
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coords_str)
                # Convert the numbers to floats and group them into pairs (x, y coordinates)
                coords_ = [(float(numbers[i]), float(numbers[i+1])) for i in range(0, len(numbers), 2)]
                # Convert the list of tuples to a 2D float array (list of lists)
                coords_2d_array = [list(coord) for coord in coords_]
                coords_array = np.array(coords_2d_array)
                # Compute the mean of the x and y coordinates
                mean_coord = np.mean(coords_array, axis=0)
                # print(atom_with_charge[str_xxx],mean_coord)
                
                # charge_dict[atom_with_charge[str_xxx]] = mean_coord
                if len(str_list) >= 2 and pre != str_xxx:
                    str = atom_with_charge[pre].replace("[", "").replace("]", "")
                    if len(str_list) >= 3:
                        if str[-1].isdigit() and str[-2] in ['+','-']:
                            tuple1, tuple2 = str_list[-2:]
                            average_coords = tuple((x1 + x2) / 2 for x1, x2 in zip(tuple1, tuple2))
                            average_coords_list.append([average_coords[0]-radius*2 , average_coords[1]-radius*2 ,average_coords[0]+radius*2 , average_coords[1]+radius*2])
                            charge = extract_charge(atom_with_charge[pre])
                            annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*4, radius*4],
                                # 'bbox_mode':   BoxMode.XYWH_ABS,
                                'category_id': charge}
                            annotations.append(annotation)
                        else:
                            average_coords = str_list[-1:][0]
                            average_coords_list.append([average_coords[0]-radius , average_coords[1]-radius,average_coords[0]+radius , average_coords[1]+radius])
                            charge = extract_charge(atom_with_charge[pre])
                            annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*2, radius*2],
                                # 'bbox_mode':   BoxMode.XYWH_ABS,
                                'category_id': charge}
                            annotations.append(annotation)
                    elif len(str_list) == 2:
                        average_coords = str_list[-1:][0]
                        average_coords_list.append([average_coords[0]-radius , average_coords[1]-radius,average_coords[0]+radius , average_coords[1]+radius])
                        charge = extract_charge(atom_with_charge[pre])
                        annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*2, radius*2],
                            # 'bbox_mode':   BoxMode.XYWH_ABS,
                            'category_id': charge}
                        annotations.append(annotation)
                    str_list = []

                str_list.append(tuple((mean_coord).tolist()))

                index = charges_ids.index(str_xxx)
                is_last_element = index == len(charges_ids) - 1
                if is_last_element and len(str_list) >= 2:
                    str = atom_with_charge[pre].replace("[", "").replace("]", "")
                    if len(str_list) >= 3:
                        if str[-1].isdigit() and str[-2] in ['+','-'] :
                            tuple1, tuple2 = str_list[-2:]
                            average_coords = tuple((x1 + x2) / 2 for x1, x2 in zip(tuple1, tuple2))
                            average_coords_list.append([average_coords[0]-radius*2 , average_coords[1]-radius*2 ,average_coords[0]+radius*2 , average_coords[1]+radius*2])
                            charge = extract_charge(atom_with_charge[pre])
                            annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*4, radius*4],
                                # 'bbox_mode':   BoxMode.XYWH_ABS,
                                'category_id': charge}
                            annotations.append(annotation)
                        else:
                            average_coords = str_list[-1:][0]
                            average_coords_list.append([average_coords[0]-radius , average_coords[1]-radius,average_coords[0]+radius , average_coords[1]+radius])
                            charge = extract_charge(atom_with_charge[pre])
                            annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*2, radius*2],
                                # 'bbox_mode':   BoxMode.XYWH_ABS,
                                'category_id': charge}
                            annotations.append(annotation)
                    elif len(str_list) == 2:
                        average_coords = str_list[-1:][0]
                        average_coords_list.append([average_coords[0]-radius , average_coords[1]-radius,average_coords[0]+radius , average_coords[1]+radius])
                        charge = extract_charge(atom_with_charge[pre])
                        annotation = {'bbox': [average_coords[0]-radius , average_coords[1]-radius, radius*2, radius*2],
                            # 'bbox_mode':   BoxMode.XYWH_ABS,
                            'category_id': charge}
                        annotations.append(annotation)
                        
                    
                
            
                print(str_list)
                pre = str_xxx
                # pre_charge_ids = charges_ids


    print(annotations)
    print(average_coords_list)



import rdkit
import numpy as np
from typing import Dict, List, Union

ALLOWABLE_FEATURES = {
    "possible_atomic_num": list(range(1, 119)),
    "possible_chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
    "possible_degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic": [False, True],
    "possible_is_in_ring": [False, True],
    "possible_bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "possible_bond_dirs": ["NONE", "ENDUPRIGHT", "ENDDOWNRIGHT"],
    "possible_bond_stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "possible_is_conjugated": [False, True],
}


def safe_index(ls: List, ele: Union[str, int, bool]):
    try:
        return ls.index(ele)
    except ValueError:
        return len(ls) - 1


def atom_to_feature_vector(atom: rdkit.Chem.rdchem.Atom):
    feature_vector = [
        safe_index(ALLOWABLE_FEATURES["possible_atomic_num"], atom.GetAtomicNum()),
        safe_index(ALLOWABLE_FEATURES["possible_chirality"], str(atom.GetChiralTag())),
        safe_index(ALLOWABLE_FEATURES["possible_degree"], atom.GetDegree()),
        safe_index(ALLOWABLE_FEATURES["possible_formal_charge"], atom.GetFormalCharge()),
        safe_index(ALLOWABLE_FEATURES["possible_numH"], atom.GetTotalNumHs()),
        safe_index(ALLOWABLE_FEATURES["possible_number_radical_e"], atom.GetNumRadicalElectrons()),
        safe_index(ALLOWABLE_FEATURES["possible_hybridization"], str(atom.GetHybridization())),
        safe_index(ALLOWABLE_FEATURES["possible_is_aromatic"], atom.GetIsAromatic()),
        safe_index(ALLOWABLE_FEATURES["possible_is_in_ring"], atom.IsInRing()),
    ]
    return feature_vector


def bond_to_feature_vector(bond: rdkit.Chem.rdchem.Bond):
    feature_vector = [
        safe_index(ALLOWABLE_FEATURES["possible_bond_type"], str(bond.GetBondType())),
        safe_index(ALLOWABLE_FEATURES["possible_bond_dirs"], str(bond.GetBondDir())),
        safe_index(ALLOWABLE_FEATURES["possible_bond_stereo"], str(bond.GetStereo())),
        safe_index(ALLOWABLE_FEATURES["possible_is_conjugated"], bond.GetIsConjugated()),
    ]
    return feature_vector


def mol_to_graph_dict(molecule: rdkit.Chem.rdchem.Mol, properties_dict: Union[Dict[str, float], None] = None):
    smiles = rdkit.Chem.MolToSmiles(molecule)
    try:
        conformer = molecule.GetConformer()
    except ValueError:
        conformer = None

    node_attr = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()]
    node_type = [atom_feature[0] for atom_feature in node_attr]
    node_chiral_type = [atom_feature[1] for atom_feature in node_attr]

    num_bond_features = 3
    if len(molecule.GetBonds()) > 0:
        edges_ls = []
        edges_features_ls = []
        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_ls.append([i, j])
            edges_features_ls.append(edge_feature)
            edges_ls.append([j, i])
            edges_features_ls.append(edge_feature)
        edge_index = np.array(edges_ls, dtype=np.int64).T.tolist()
        edge_attr = np.array(edges_features_ls, dtype=np.int64).tolist()
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64).tolist()
        edge_attr = np.zeros((0, num_bond_features), dtype=np.int64).tolist()
    edge_type = [edge_feature[0] for edge_feature in edge_attr]
    edge_dir_type = [edge_feature[1] for edge_feature in edge_attr]
    graph = {
        "node_type": node_type,
        "node_chiral_type": node_chiral_type,
        "edge_type": edge_type,
        "edge_dire_type": edge_dir_type,
        "edge_index": edge_index,
    }
    if conformer is not None:
        graph["conformer"] = conformer.GetPositions().tolist()
    graph.update(properties_dict) if properties_dict is not None else None

    return graph

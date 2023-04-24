import rdkit
import rdkit.Chem as Chem
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from vocab import *
import pprint
import json

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands,aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = zip(*cands)
            self.cands = list(self.cands)
        else:
            self.cands = []

class MolTree(object):

    def __init__(self, smiles):
        # print('MolTree init. smiles: ', smiles)
        self.smiles = smiles
        self.nodes = []
        self.n_errors = 0

        self.mol = get_mol(smiles)

        if (self.mol is None):
            self.n_errors += 1
            return

        #Stereo Generation (currently disabled)
        #mol = Chem.MolFromSmiles(smiles)
        #self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        #self.smiles2D = Chem.MolToSmiles(mol)
        #self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol)
        root = 0
        # print('cliques: ', json.dumps(cliques, indent=2))

        for i,c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            if cmol is None:
                self.n_errors += 1
                node = None
            else:
                node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0 and cmol is not None: root = i

        # print('self.n_errors: ', self.n_errors)
        for x,y in edges:
            if self.nodes[x] is not None and self.nodes[y] is not None:
               # print('Adding edge.')
               self.nodes[x].add_neighbor(self.nodes[y])
               self.nodes[y].add_neighbor(self.nodes[x])
            else:
               # print('Skipping edge.')
               pass

        if root > 0:
            self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]

        for i,node in enumerate(self.nodes):
            if node is None:
               continue
            node.nid = i + 1
            if len(node.neighbors) > 1: #Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            if node is not None:
                node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            if node is not None:
                node.assemble()

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx: continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    total_invalid_mols = 0
    total_mols = 0
    cset = set()
    for line in sys.stdin:
        smiles_list = line.split()
        if len(smiles_list) < 1:
            continue
        smiles = smiles_list[0]
        total_mols += 1
        mol = MolTree(smiles)
        if mol is None or mol.n_errors > 0:
            total_invalid_mols += 1
            # Print 5 invalid entries every 1000 ones.
            if total_invalid_mols % 1000 < 5:
                sys.stderr.write(f'Total Mols: {total_mols}; Total Invalids: {total_invalid_mols}')
            continue
        for c in mol.nodes:
            if c is not None:
               cset.add(c.smiles)
    for x in cset:
        print(x)


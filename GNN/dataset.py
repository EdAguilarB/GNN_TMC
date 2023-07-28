import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
from coords_composition_pbc import *
import os.path as osp
import os
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale


###node features atom identity and coords.Edge features distance between nodes
class MoC_c_dist(Dataset):

    def __init__(self, root, max_d = 3.2, max_n = 26, lattice = 8.54, transform=None, pre_transform=None, pre_filter=None):

        self.max_d = max_d
        self.max_n = max_n
        self.lattice = lattice
        energy = pd.read_csv('/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Data/E_data/cell.features.dat', sep= ' ', usecols=[0])
        energy = energy.to_numpy()
        energy = (energy-np.min(energy))*13.605693122994
        self.energy = torch.tensor(energy, dtype=float)

        super(MoC_c_dist, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return [f'{i}.vasp' for i in range(1,201)]

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt' for idx in range(1,201)]

    def download(self):
        pass

    def process(self):
        idx = 1
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path, header=5)
            df = df.rename(columns = {df.columns[0]: 'Mo C'})

            node_feats = self._get_node_feats(df)
            edge_feats = self._get_edge_feats(df)
            adj = self._get_adj(df)
            energy = self._get_energy(idx)
            file_name = f'{idx}.vasp'
            print(file_name)
            data = Data(file = file_name,
                        x=node_feats,
                        edge_index=adj,
                        edge_attr=edge_feats,
                        y=energy)

            torch.save(data, osp.join(self.processed_dir, f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt'))
            idx += 1

    def _get_node_feats(self, df):
        composition = get_composition(df)
        Mo_y = [1 for _ in range(composition['Mo'][0])]
        Mo_n = [0 for _ in range(composition['C'][0])]
        Mo = Mo_y + Mo_n
        C_n = [Mo_y[i] - 1 for i in range(len(Mo_y))]
        C_y = [Mo_n[i] + 1 for i in range(len(Mo_n))]
        C = C_n + C_y
        Mo = np.expand_dims(np.asarray(Mo), axis = 1)
        C = np.expand_dims(np.asarray(C), axis = 1)
        feat = np.concatenate([Mo, C], axis = 1)
        feat = torch.tensor(feat, dtype=torch.float)
        coords = get_coords(df)
        coords = torch.tensor(coords)
        feat = torch.concat((feat, coords), axis = 1)
        return feat
    
    def _get_adj(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        coo = np.array(list(zip(row,col)))
        coo = np.transpose(coo)
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_edge_feats(self, df):
        edge_dist = []
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)

        for i in range(len(row)):
            r = row[i]
            c = col[i]
            d = distances[r,c]
            edge_dist.append(d)

        edge_dist = np.asarray(edge_dist)
        edge_dist = torch.tensor(edge_dist, dtype=torch.float)
        return edge_dist.unsqueeze(1)
    
    
    def _get_energy(self, idx):
        e = self.energy[idx-1]
        return e
    

    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'processed_{idx+1}_mn_{self.max_n}_md_{self.max_d}.pt'))
        return data
    

#node features atom identity. Edge features direction
class MoC_dir(Dataset):

    def __init__(self, root, max_d = 3.2, max_n = 26, lattice = 8.54, transform=None, pre_transform=None, pre_filter=None):

        self.max_d = max_d
        self.max_n = max_n
        self.lattice = lattice
        energy = pd.read_csv('/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Data/E_data/cell.features.dat', sep= ' ', usecols=[0])
        energy = energy.to_numpy()
        energy = (energy-np.min(energy))*13.605693122994
        self.energy = torch.tensor(energy, dtype=float)

        super(MoC_dir, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return [f'{i}.vasp' for i in range(1,201)]

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt' for idx in range(1,201)]

    def download(self):
        pass

    def process(self):
        idx = 1
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path, header=5)
            df = df.rename(columns = {df.columns[0]: 'Mo C'})

            node_feats = self._get_node_feats(df)
            edge_feats = self._get_edge_feats(df)
            adj = self._get_adj(df)
            energy = self._get_energy(idx)

            data = Data(x=node_feats,
                        edge_index=adj,
                        edge_attr=edge_feats,
                        y=energy)

            torch.save(data, osp.join(self.processed_dir, f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt'))
            idx += 1

    def _get_node_feats(self, df):
        composition = get_composition(df)
        Mo_y = [1 for _ in range(composition['Mo'][0])]
        Mo_n = [0 for _ in range(composition['C'][0])]
        Mo = Mo_y + Mo_n
        C_n = [Mo_y[i] - 1 for i in range(len(Mo_y))]
        C_y = [Mo_n[i] + 1 for i in range(len(Mo_n))]
        C = C_n + C_y
        Mo = np.expand_dims(np.asarray(Mo), axis = 1)
        C = np.expand_dims(np.asarray(C), axis = 1)
        feat = np.concatenate([Mo, C], axis = 1)
        feat = torch.tensor(feat, dtype=torch.float)
        return feat
    
    def _get_adj(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        coo = np.array(list(zip(row,col)))
        coo = np.transpose(coo)
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_edge_feats(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        edge_dir = np.zeros((len(row), 6))

        for i in range(len(row)):
            r = row[i]
            c = col[i]
            c1 = coords[r]
            c2 = coords[c]
            edge_dir[i] = edge_features_pbc(c1, c2, self.lattice)

        edge_feats = torch.tensor(edge_dir, dtype=torch.float)
        return edge_feats
    
    
    def _get_energy(self, idx):
        e = self.energy[idx-1]
        return e
    

    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'processed_{idx+1}_mn_{self.max_n}_md_{self.max_d}.pt'))
        return data



###node features atom identity and coords.Edge features direction of the edge
class MoC_c_dir(Dataset):

    def __init__(self, root, max_d = 3.2, max_n = 26, lattice = 8.54, transform=None, pre_transform=None, pre_filter=None):

        self.max_d = max_d
        self.max_n = max_n
        self.lattice = lattice
        energy = pd.read_csv('/Users/ed_aguilar/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/NewData/Data/energy_descriptors_all.csv', sep=',',index_col=0)
        energy['dft'] = (energy['dft']-min(energy['dft']))*13.605693122994
        self.energy = energy['dft']

        super(MoC_c_dir, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return os.listdir('/Users/ed_aguilar/Documents/PhD_Documents/Experiments/Metal_carbides/Experiments_periodicity/NewData/Data/raw')

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt' for idx in range(1,len(self.raw_file_names))]

    def download(self):
        pass

    def process(self):
        idx = 1
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            try:
                df = pd.read_csv(raw_path, header=5)
                df = df.rename(columns = {df.columns[0]: 'Mo C'})

                node_feats = self._get_node_feats(df)
                edge_feats = self._get_edge_feats(df)
                adj = self._get_adj(df)
                energy = self._get_energy(raw_path)
                file_idx = f'{idx}'

                data = Data(x=node_feats,
                            edge_index=adj,
                            edge_attr=edge_feats,
                            y=energy,
                            file_idx = file_idx,
                            file_name = raw_path.split('/')[-1])

                torch.save(data, osp.join(self.processed_dir, f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt'))
                idx += 1
            except:
                print(raw_path)

    def _get_node_feats(self, df):
        composition = get_composition(df)
        Mo_y = [1 for _ in range(composition['Mo'][0])]
        Mo_n = [0 for _ in range(composition['C'][0])]
        Mo = Mo_y + Mo_n
        C_n = [Mo_y[i] - 1 for i in range(len(Mo_y))]
        C_y = [Mo_n[i] + 1 for i in range(len(Mo_n))]
        C = C_n + C_y
        Mo = np.expand_dims(np.asarray(Mo), axis = 1)
        C = np.expand_dims(np.asarray(C), axis = 1)
        feat = np.concatenate([Mo, C], axis = 1)
        feat = torch.tensor(feat, dtype=torch.float)
        coords = get_coords(df)
        coords = torch.tensor(coords, dtype=float)
        feat = torch.concat((feat, coords), axis = 1)
        return feat
    
    def _get_adj(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        coo = np.array(list(zip(row,col)))
        coo = np.transpose(coo)
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_edge_feats(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        edge_dir = np.zeros((len(row), 6))

        for i in range(len(row)):
            r = row[i]
            c = col[i]
            c1 = coords[r]
            c2 = coords[c]
            edge_dir[i] = edge_features_pbc(c1, c2, self.lattice)

        edge_feats = torch.tensor(edge_dir, dtype=torch.float)
        return edge_feats
    
    
    def _get_energy(self, raw_path):
        structure = raw_path.split('/')[-1]
        e = [self.energy.loc[structure]]
        return torch.tensor(e, dtype=float)
    

    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'processed_{idx+1}_mn_{self.max_n}_md_{self.max_d}.pt'))
        return data
    
    

###node features atom identity and coords. Edge features distance between nodes and direction
class MoC_c_dist_dir(Dataset):

    def __init__(self, root, max_d = 3.2, max_n = 26, lattice = 8.54, transform=None, pre_transform=None, pre_filter=None):

        self.max_d = max_d
        self.max_n = max_n
        self.lattice = lattice
        energy = pd.read_csv('/Users/ed/Documents/PhD_Documents/Experiments/Metal_carbides/Data/E_data/cell.features.dat', sep= ' ', usecols=[0])
        energy = energy.to_numpy()
        energy = (energy-np.min(energy))*13.605693122994
        self.energy = torch.tensor(energy, dtype = float)

        super(MoC_c_dist_dir, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return [f'{i}.vasp' for i in range(1,201)]

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt' for idx in range(1,201)]

    def download(self):
        pass

    def process(self):
        idx = 1
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path, header=5)
            df = df.rename(columns = {df.columns[0]: 'Mo C'})

            node_feats = self._get_node_feats(df)
            edge_feats = self._get_edge_feats(df)
            adj = self._get_adj(df)
            energy = self._get_energy(idx)

            data = Data(x=node_feats,
                        edge_index=adj,
                        edge_attr=edge_feats,
                        y=energy)

            torch.save(data, osp.join(self.processed_dir, f'processed_{idx}_mn_{self.max_n}_md_{self.max_d}.pt'))
            idx += 1

    def _get_node_feats(self, df):
        composition = get_composition(df)
        Mo_y = [1 for _ in range(composition['Mo'][0])]
        Mo_n = [0 for _ in range(composition['C'][0])]
        Mo = Mo_y + Mo_n
        C_n = [Mo_y[i] - 1 for i in range(len(Mo_y))]
        C_y = [Mo_n[i] + 1 for i in range(len(Mo_n))]
        C = C_n + C_y
        Mo = np.expand_dims(np.asarray(Mo), axis = 1)
        C = np.expand_dims(np.asarray(C), axis = 1)
        feat = np.concatenate([Mo, C], axis = 1)
        feat = torch.tensor(feat, dtype=torch.float)
        coords = torch.tensor(get_coords(df))
        feat = torch.concat((feat, coords), axis = 1)
        return feat
    
    def _get_adj(self, df):
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        coo = np.array(list(zip(row,col)))
        coo = np.transpose(coo)
        return torch.tensor(coo, dtype=torch.long)
    
    def _get_edge_feats(self, df):
        edge_dist = []
        coords = get_coords(df)
        distances = get_distances_pbc(coords, self.lattice)
        adj = get_adj_matrix(distances, max_neighbours = self.max_n, max_distance = self.max_d)
        row,col = np.where(adj)
        edge_dir = np.zeros((len(row), 6))

        for i in range(len(row)):
            r = row[i]
            c = col[i]
            d = distances[r,c]

            c1 = coords[r]
            c2 = coords[c]
            edge_dir[i] = edge_features_pbc(c1, c2, self.lattice)

            edge_dist.append(d)

        edge_dist = np.asarray(edge_dist)
        edge_dist = np.expand_dims(edge_dist, axis=1)
        edge_feats = np.concatenate([edge_dist, edge_dir], axis = 1)
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)
        return edge_feats
    
    
    def _get_energy(self, idx):
        e = self.energy[idx-1]
        return e
    

    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'processed_{idx+1}_mn_{self.max_n}_md_{self.max_d}.pt'))
        return data


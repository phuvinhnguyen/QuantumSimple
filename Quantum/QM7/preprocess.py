import scipy.io
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import ConcatDataset
import os

WORKING_DIR = os.getcwd()

def create_graph(coulomb_matrix, energy):
    num_atoms = coulomb_matrix.shape[0]
    edge_index = torch.tensor([[i, j] for i in range(num_atoms) for j in range(num_atoms)], dtype=torch.long).t().contiguous()
    x = torch.tensor(np.diag(coulomb_matrix), dtype=torch.float).view(-1, 1)
    edge_attr = torch.tensor([coulomb_matrix[i, j] for i in range(num_atoms) for j in range(num_atoms)], dtype=torch.float).view(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([energy], dtype=torch.float))
    return data

def create_dataloader_list(
        coulomb_matrix,
        atomization_energy,
        split,
        batch_size,
        shuffle,
        donot_loader=False,
        ):
    data_split = coulomb_matrix[split]
    atomization_energy_split = atomization_energy[split]
    datasets = [[create_graph(data[i], atom[i]) for i in range(data.shape[0])] for data, atom in zip(data_split, atomization_energy_split)]

    if donot_loader:
        test_dataset = datasets.pop(-1)
    else:
        test_dataset = DataLoader(datasets.pop(-1), batch_size=batch_size, shuffle=shuffle)
    
    train_dataset = []
    validation_dataset = []

    for index in range(len(datasets)):
        if donot_loader:
            validation_dataset.append(datasets[index])
            train_dataset.append(ConcatDataset(
                [datasets[i] for i in range(len(datasets)) if i != index]
                ))
        else:
            validation_dataset.append(DataLoader(datasets[index], batch_size=batch_size, shuffle=shuffle))
            train_dataset.append(
                DataLoader(ConcatDataset(
                    [datasets[i] for i in range(len(datasets)) if i != index]
                ), batch_size=batch_size, shuffle=shuffle)
            )

    return {
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
    }

def load_dataset(batch_size=32, shuffle=True, donot_loader=False):
    data = scipy.io.loadmat(f'{WORKING_DIR}/Quantum/QM7/data/qm7.mat')
    coulomb_matrices = data['X']
    atomization_energies = data['T'].flatten()

    return create_dataloader_list(
        coulomb_matrices,
        atomization_energies,
        data['P'],
        batch_size,
        shuffle,
        donot_loader=donot_loader,
        )
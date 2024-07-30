import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

def run_example(
        model,
        core_data,
        device='cpu',
        index_sample: dict = {
            'train': (0, 2),
            'validation': (0, 0),
            'test': 2
        },
        ):
    print('Start running some examples ...')
    model.eval()
    result = []

    with torch.no_grad():
        for name, index in index_sample.items():
            if isinstance(index, int):
                example_data = core_data[name][index].to(device)
            else:
                example_data = core_data[name][index[0]][index[1]].to(device)
            print(f'Example: Predict the atomization energy of a molecule of sample {index} in set {name}.')
            example_output = model(example_data)
            print(f'Predicted Atomization Energy: {example_output.item():.2f}')
            print(f'Actual Atomization Energy: {example_data.y.item():.2f}')
            print()
            
            result.append(example_data)
    
    return result

def eval_old(
        model,
        loader,
        core_data,
        device='cpu',
        ):
    model.eval()
    error = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            error += F.l1_loss(output, data.y).item() * data.num_graphs

    print(f'Error: {error / len(loader.dataset):.4f}')

    example_result = run_example(model, core_data, device)

    return example_result

def eval(
        model,
        loader,
        core_data,
        device='cpu',
        ):
    model.eval()
    mse_error = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            mse_error += F.mse_loss(output, data.y, reduction='sum').item()  # Sum MSE loss for each batch
            all_outputs.append(output.cpu())
            all_targets.append(data.y.cpu())

    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    r2 = r2_score(all_targets, all_outputs)

    print(f'Test result: MSE = {mse_error / len(loader.dataset):.4f}, R2 = {r2:.4f}')

    example_result = run_example(model, core_data, device)

    return example_result

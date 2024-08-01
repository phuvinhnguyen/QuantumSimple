import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def test_(model, loader, device='cpu'):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            all_outputs.append(output.cpu())
            all_targets.append(data.y.cpu())

    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mae_error = mean_absolute_error(all_targets, all_outputs)

    return mae_error

def train_(
        model,
        loader,
        device='cpu',
        optimizer=None,
        criterion=torch.nn.MSELoss()
        ):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)

def train(
        model,
        train_loader_list,
        eval_loader_list,
        optimizer=None,
        criterion=None,
        lr=0.001,
        epochs=201,
        save_path=None,
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
    criterion = torch.nn.MSELoss() if criterion is None else criterion

    k_fold_number = len(train_loader_list)
    min_val_loss = 9999999999999999

    for epoch in range(1, epochs):
        train_loss = train_(model, train_loader_list[epoch%k_fold_number], device, optimizer, criterion)
        val_loss = test_(model, eval_loader_list[epoch%k_fold_number], device)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            if val_loss < min_val_loss and save_path is not None:
                min_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
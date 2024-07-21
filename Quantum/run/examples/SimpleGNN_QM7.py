from ..train.DeepTrain import train
from ..eval.DeepEval import eval
from ...QM7.preprocess import load_dataset
from ...models.GNN import GNN_regresion
from ...tools import timeit

@timeit
def run(
        epochs=201,
        lr=0.001,
        batch_size=32,
        criterion=None,
        optimizer=None,
        save_path=None,
    ):
    datasets = load_dataset(batch_size, True)
    core_data = load_dataset(batch_size, False, True)
    train_datasets = datasets['train']
    eval_datasets = datasets['validation']
    test_dataset = datasets['test']

    model = GNN_regresion()

    train(
        model,
        train_loader_list=train_datasets,
        eval_loader_list=eval_datasets,
        optimizer=optimizer,
        epochs=epochs,
        criterion=criterion,
        lr=lr,
        save_path=save_path
        )
    
    _ = eval(model, loader=test_dataset, core_data=core_data, device='cpu')

    return model
from ...models.GP import Gaussian_Processes
from ...tools import timeit
from ..train.MLTrain import train
from torch.utils.data import ConcatDataset
from ..eval.MLEval import eval
from ...QM7.preprocess import load_dataset

@timeit
def run():
    dataset = load_dataset(donot_loader=True)
    data = ConcatDataset([dataset['train'][0], dataset['validation'][0]])
    test_data = dataset['test']
    model = Gaussian_Processes()

    train(model, data)
    eval(model, test_data, data)

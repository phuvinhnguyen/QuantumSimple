from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..train.MLTrain import preprocess_data

def run_example(
    model,
    test_data,
    train_data,
    index_sample: dict = {
        'train': [2,3],
        'test': [2,3]
    },
    ):
    print('Start running some examples ...')
    for name, index in index_sample.items():
        if name == 'train':
            X = train_data[0][index]
            Y = train_data[1][index]
        else:
            X = test_data[0][index]
            Y = test_data[1][index]
        print(f'Example: Predict the atomization energy of a molecule of sample {index} in set {name}.')
        example_output = model.predict(X)
        print(f'Predicted Atomization Energy: {example_output}')
        print(f'Actual Atomization Energy: {Y}')
        print()

def eval(
    model,
    test_data,
    train_data,
    preprocess_data=preprocess_data,
    ):
    X_test, Y_test = preprocess_data(test_data)
    X_train, Y_train = preprocess_data(train_data)

    Y_predict = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)

    print(f'Test result: MSE = {mse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}')

    run_example(model, (X_test, Y_test), (X_train, Y_train))
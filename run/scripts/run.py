from ..examples.GP_QM7 import run as GP_run
from ..examples.SVR_QM7 import run as SVR_run
from ..examples.SimpleGNN_QM7 import run as SimpleGNN_run
from ..examples.linear_QM7 import run as linear_run
from ..examples.mlp_QM7 import run as mlp_run
from ..examples.XGBoost_QM7 import run as XGBoost_run
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', type=str, required=True, help='method to run')
    return parser.parse_args()

def main():
    args = parse_args()
    
    method_name = args.input

    if method_name == 'GP':
        GP_run()
    elif method_name == 'SVR':
        SVR_run()
    elif method_name == 'SimpleGNN':
        SimpleGNN_run()
    elif method_name == 'linear':
        linear_run()
    elif method_name == 'mlp':
        mlp_run()
    elif method_name == 'XGBoost':
        XGBoost_run()
    elif method_name == 'all':
        GP_run()
        SVR_run()
        SimpleGNN_run()
        linear_run()
        mlp_run()
        XGBoost_run()
    else:
        print('''
        Wrong method name.
        Available methods: GP, SVR, SimpleGNN, linear, mlp
        ''')

if __name__ == "__main__":
    main()
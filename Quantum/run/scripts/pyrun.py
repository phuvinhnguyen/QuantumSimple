from ..examples.GP_QM7 import run as GP_run
from ..examples.SVR_QM7 import run as SVR_run
from ..examples.SimpleGNN_QM7 import run as SimpleGNN_run
from ..examples.linear_QM7 import run as linear_run
from ..examples.mlp_QM7 import run as mlp_run
from ..examples.GAT_QM7 import run as GAT_run
from ..examples.XGBoost_QM7 import run as XGBoost_run

def run(method_name='all'):
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
        print('*'*50)
        print('Running GP')
        GP_run()
        print('*'*50)

        print('Running SVR')
        SVR_run()
        print('*'*50)

        print('Running SimpleGNN')
        SimpleGNN_run()
        print('*'*50)

        print('Running linear')
        linear_run()
        print('*'*50)

        print('Running mlp')
        mlp_run()
        print('*'*50)

        print('Running XGBoost')
        XGBoost_run()
        print('*'*50)

        print('Running GAT')
        GAT_run()
        print('*'*50)
    else:
        print('''
        Wrong method name.
        Available methods: GP, SVR, SimpleGNN, linear, mlp
        ''')
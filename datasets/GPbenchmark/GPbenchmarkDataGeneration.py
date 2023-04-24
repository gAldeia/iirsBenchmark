import pandas as pd
import numpy  as np

benchmarks = {
    'Korns-11'        : lambda x, y, z, v, w: 6.87 + 11*np.cos( 7.23*(x**3) ),
    'Korns-12'        : lambda x, y, z, v, w: 2 - 2.1*np.cos(9.8*x)*np.sin(1.3*w),
    'Vladislavleva-4' : lambda x1, x2, x3, x4, x5: 10/(5 + sum([(x - 3)**2 for x in [x1, x2, x3, x4, x5]])),
    'Pagie-1'         : lambda x, y: 1/(1 + x**-4) + 1/(1 + y**-4)
}

def U(a, b, c):
    epsilon = 1e-6
    while True:
        yield np.random.uniform(a, b+epsilon, size=c)
    
    
def E(a, b, c):
    while True:
        arr = np.arange(a, b+c, step=c)
        np.random.shuffle(arr)

        yield arr

    
var_names = {
    'Korns-11'        : ('x', 'y', 'z', 'v', 'w'),
    'Korns-12'        : ('x', 'y', 'z', 'v', 'w'),
    'Vladislavleva-4' : ('x1', 'x2', 'x3', 'x4', 'x5'),
    'Pagie-1'         : ('x', 'y'),
}
    
training_sampling = {
    'Korns-11'        : U(-50., 10., 1000),
    'Korns-12'        : U(-50., 10., 1000),
    'Vladislavleva-4' : U(0.05, 6.05, 1000),
    'Pagie-1'         : E(-5.0, 5.01, 0.01),
}

testing_sampling = {
    'Korns-11'        : U(-50., 10., 100),
    'Korns-12'        : U(-50., 10., 100),
    'Vladislavleva-4' : U(-0.5, 10.05, 100),
    'Pagie-1'         : E(-5.0, 5.1, 0.1),
}

if __name__ == '__main__':
    for name, f in benchmarks.items():
        print(name)

        data_train = pd.DataFrame()
        data_test  = pd.DataFrame()

        #filling train and test Xs
        for i, var_name in enumerate(var_names[name]):
            print(i, var_name)
            data_train[var_name] = next(training_sampling[name])
            data_test[var_name] = next(testing_sampling[name])

        # evaluating ys
        data_train['f(args...)'] = data_train.apply(lambda row: f(*row.to_numpy()), axis=1)
        data_test['f(args...)']  = data_test.apply(lambda row: f(*row.to_numpy()), axis=1)

        print(data_train.shape)
        print(data_test.shape)

        data_train.to_csv(f'train/{name}_UNI.csv', index=False)
        data_test.to_csv(f'test/{name}_UNI.csv', index=False)
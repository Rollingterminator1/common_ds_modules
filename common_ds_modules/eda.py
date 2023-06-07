import pandas as pd
import matplotlib.pyplot as plt

def plot_binned_var(train, continuous_variables):
    num_bins = 5
    labels = [i + 1 for i in range(0, num_bins)]
    for n in continuous_variables:
        std = train[n].std()
        if std > 10:
            train[f'{n}_{num_bins}_bins'] = pd.cut(train[n], num_bins,
                                                                       labels=labels)
            print(f'Variable: {n}, std: {std}')
            plt.hist(train[f'{n}_{num_bins}_bins'])
            plt.title(f'Distribution for {n}_{num_bins}_bins')
            plt.show()

def plot_variable_dist(df, variables):
    for c in variables:
        plt.hist(df[c])
        plt.title(f'Distribution for {c}')
        plt.show()

#plot_variable_dist(train_df, highly_correlated_variables)
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
import numpy as np

def make_continuous_plot(feature,train_detail):
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(2, 2)

    j = sns.scatterplot(y=train_detail['Weekly_Sales'],
                        x=boxcox1p(train_detail[feature], 0.15),
                        ax=fig.add_subplot(gs[0, 1]),
                        palette='blue')

    plt.title('BoxCox 0.15\n' + 'Corr: ' + str(
        np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.15)), 2)) +
              ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.15), nan_policy='omit'), 2)))

    j = sns.scatterplot(y=train_detail['Weekly_Sales'],
                        x=boxcox1p(train_detail[feature], 0.25),
                        ax=fig.add_subplot(gs[1, 0]),
                        palette='blue')
    plt.title('BoxCox 0.25\n' + 'Corr: ' +
              str(np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.25)), 2)) +
              ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.25), nan_policy='omit'), 2)))

    j = sns.distplot(train_detail[feature], ax=fig.add_subplot(gs[1, 1]), color='green')

    plt.title('Distribution\n')

    j = sns.scatterplot(y=train_detail['Weekly_Sales'],
                        x=train_detail[feature],
                        ax=fig.add_subplot(gs[0, 0]),
                        color='red')

    plt.title('Linear\n' + 'Corr: ' + str(
        np.round(train_detail['Weekly_Sales'].corr(train_detail[feature]), 2)) + ', Skew: ' +
              str(np.round(stats.skew(train_detail[feature], nan_policy='omit'), 2)))

    fig.show()
    



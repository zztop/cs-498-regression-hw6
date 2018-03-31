import matplotlib.pyplot as plt
import pandas as pd
from glmnet import LogitNet, ElasticNet
import numpy as np
from sklearn.linear_model import ElasticNetCV
import scipy


def run_logistic_regression(run_alpha):
    m = ElasticNetCV(l1_ratio=run_alpha, n_alphas=20, cv=20)  # scoring defaults to classification error
    m.fit(credit_features, default_label)
    mean_accuracy = m.score(credit_features, default_label)
    # plot_log_regularized_values(m, 'img.png')
    print('Mean accuracy for alpha {} {}'.format(run_alpha, mean_accuracy))
    return m


def plot_log_regularized_values(image_name, ms=[]):
    plt.figure()
    # plt.plot(feature, label, 'r.', markersize=12)
    for m in ms:
        plt.plot(scipy.log(m.lambda_path_), m.cv_mean_score_, markersize=8,
                 label='alpha {}'.format(str(m.alpha)))

    plt.legend()
    plt.xlabel("log lambda ")
    plt.ylabel("mean squared error")

    plt.savefig(image_name)
    plt.close('all')


if __name__ == '__main__':
    all_data = pd.read_excel('./default of credit card clients.xls', header=[0])

    default_label = all_data.iloc[1:all_data.shape[0], all_data.shape[1] - 1].astype('int').values.reshape(-1)
    credit_features = all_data.iloc[1:all_data.shape[0], 0:all_data.shape[1] - 1].values
    # ridge
    # alpha = 0
    # m0 = run_logistic_regression(alpha)
    # lasso
    alpha = 1
    m1 = run_logistic_regression(alpha)

    # alpha =0.1
    alpha = 0.1
    m_dot1 = run_logistic_regression(alpha)
    # alpha =0.5
    # alpha = 0.5
    # m_dot5 = run_logistic_regression(alpha)
    # # alpha =0.8
    # alpha = 0.8
    # m_dot8 = run_logistic_regression(alpha)
    # plot_log_regularized_values('all_m_comparison.png', [m0, m1, m_dot1, m_dot5, m_dot8])
    plot_log_regularized_values('all_m_comparison.png', [ m1, m_dot1])
    print('done')

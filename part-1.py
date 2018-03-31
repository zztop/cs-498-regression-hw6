import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glmnet import ElasticNet
from scipy import stats, special
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import cross_val_predict


def inverse_box_cox(y, ld, additive):
    return special.inv_boxcox(y, ld) - additive


def plot_predictions(predictions, label, image_name, label_name):
    plt.figure()
    plt.plot(predictions, label - predictions, 'r.')

    slope, intercept = np.polyfit(predictions, label - predictions, 1)
    abline_values = [slope * i + intercept for i in predictions]
    plt.plot(predictions, abline_values, 'b')
    plt.xlabel("fitted ")
    plt.ylabel("residual")
    plt.title(label_name)

    # plt.show()
    plt.savefig(image_name)
    plt.close('all')


def l1_l2_regression(alpha):
    m = ElasticNet(n_splits=20, scoring='r2', alpha=alpha)
    m.fit(music_features, box_latitude_label)
    lat_r_squared = m.score(music_features, box_latitude_label)
    print('GLMNET L1 L2 alpha {} latitude r2 {}'.format(alpha, lat_r_squared))
    plot_predictions(inverse_box_cox(m.predict(music_features), lambda_lat, 90), latitude_label,
                     'l1_l2_latitude_residual_{}.png'.format(alpha),
                     'residual vs fitted latitude for l1_l2 \n regression alpha {}'.format(alpha))
    m.fit(music_features, box_longitude_label)
    lon_r_squared = m.score(music_features, box_longitude_label)
    print('GLMNET L1 L2 alpha {} longitude r2 {}'.format(alpha, lon_r_squared))
    plot_predictions(inverse_box_cox(m.predict(music_features), lambda_lon, 180), longitude_label,
                     'l1_l2_longitude_residual_{}.png'.format(alpha),
                     'residual vs fitted longitude for l1_l2 \n regression alpha {}'.format(alpha))


def linear_regression():
    # global latitude_r_squared, longitude_r_squared
    m = LinearRegression()
    m.fit(music_features, latitude_label)
    latitude_r_squared = m.score(music_features, latitude_label)
    print('Linear regression latitude r2 {}'.format(latitude_r_squared))
    plot_predictions(cross_val_predict(m, music_features, latitude_label,
                                       cv=20), latitude_label, 'latitude_residual.png', 'residual vs fitted latitude')
    m.fit(music_features, longitude_label)
    longitude_r_squared = m.score(music_features, longitude_label)
    print('Linear regression longitude r2 {}'.format(longitude_r_squared))
    plot_predictions(cross_val_predict(m, music_features, longitude_label,
                                       cv=20), longitude_label, 'longitude_residual.png',
                     'residual vs fitted longitude')


def box_cox_regression():
    m = LinearRegression()
    m.fit(music_features, box_latitude_label)
    lat_r_squared = m.score(music_features, box_latitude_label)
    print('Box Cox Linear regression latitude r2 {}'.format(lat_r_squared))
    plot_predictions(
        inverse_box_cox(m.predict(music_features), lambda_lat, 90),
        latitude_label,
        'box_latitude_residual.png',
        'residual vs fitted boc cox latitude')
    m.fit(music_features, box_longitude_label)
    lon_r_squared = m.score(music_features, box_longitude_label)
    print('Box Cox Linear regression longitude r2 {}'.format(lon_r_squared))
    plot_predictions(
        inverse_box_cox(m.predict(music_features), lambda_lon, 180),
        longitude_label,
        'box_longitude_residual.png',
        'residual vs fitted boxcox longitude')


def glmnet_box():
    m1 = ElasticNet(n_splits=20, scoring='r2', alpha=0)
    m1.fit(music_features, box_latitude_label)
    lat_r_squared = m1.score(music_features, box_latitude_label)
    print('GLMNET ridge lattitude r2 {}'.format(lat_r_squared))
    plot_predictions(inverse_box_cox(m1.predict(music_features), lambda_lat, 90), latitude_label,
                     'ridge_latitude_residual.png',
                     'residual vs fitted latitude for Ridge')
    m1.fit(music_features, box_longitude_label)
    lon_r_squared = m1.score(music_features, box_longitude_label)
    print('GLMNET ridge longitude r2 {}'.format(lon_r_squared))
    plot_predictions(inverse_box_cox(m1.predict(music_features), lambda_lon, 180), longitude_label,
                     'ridge_longitude_residual.png',
                     'residual vs fitted longitude for Ridge regression')


def glmnet_lasso():
    m = ElasticNet(n_splits=20, scoring='r2', alpha=1)
    m.fit(music_features, box_latitude_label)
    latitude_r_squared = m.score(music_features, box_latitude_label)
    print('GLMNET lasso latitude r2 {}'.format(latitude_r_squared))
    plot_predictions(inverse_box_cox(m.predict(music_features), lambda_lat, 90), latitude_label,
                     'lasso_latitude_residual.png',
                     'residual vs fitted latitude for lasso regression')
    m.fit(music_features, box_longitude_label)
    longitude_r_squared = m.score(music_features, box_longitude_label)
    print('GLMNET lasso longitude r2 {}'.format(longitude_r_squared))
    plot_predictions(inverse_box_cox(m.predict(music_features), lambda_lon, 180), longitude_label,
                     'lasso_longitude_residual.png',
                     'residual vs fitted longitude for lasso regression')


if __name__ == '__main__':
    all_data = pd.read_csv('./Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt',
                           names=None,
                           na_values=['?'], sep=',')

    latitude_label = all_data.iloc[:, all_data.shape[1] - 2:all_data.shape[1] - 1].values.reshape(-1)
    longitude_label = all_data.iloc[:, all_data.shape[1] - 1].values.reshape(-1)
    music_features = all_data.iloc[:, 0:all_data.shape[1] - 2].values

    linear_regression()

    box_latitude_label, lambda_lat = stats.boxcox(latitude_label + 90)
    box_longitude_label, lambda_lon = stats.boxcox(longitude_label + 180)

    box_cox_regression()

    # GLMNET For Ridge
    glmnet_box()

    # GLMNET for lasso
    glmnet_lasso()

    # GLMNET for variable alphas
    l1_l2_regression(0.1)
    l1_l2_regression(0.5)
    l1_l2_regression(0.8)

    print('done')

import datetime
from argparse import ArgumentParser
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import sys
import os
from astropy.table import QTable
from astropy.io import ascii

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))


def Gauss_resample(x, y, N):
    """
    Resample features based on Gaussian approximation

    Note we divide the covariance by 2!

    Parameters
    ----------
    X : numpy.ndarray
        Feature array
    y : numpy.ndarray
        Label array
    N : int
        Total samples to simulate (to be added to original sample)

    Returns
    -------
    newX : numpy.ndarray
        New Feature array
    newY : numpy.ndarray
        New label array
    """
    uys = np.unique(y)
    newX = np.zeros((int(N*len(uys)), np.size(x, axis=1)))
    newy = np.zeros((int(N*len(uys)), ))
    for i, uy in enumerate(uys):
        gind = np.where(y == uy)
        newX[i*N:i*N+len(gind[0]), :] = x[gind[0], :]
        newy[i*N:(i+1) * N] = uy
        cx = x[gind[0], :]
        mean = np.mean(cx, axis=0)
        cov = np.cov(cx, rowvar=False)
        newX[i * N + len(gind[0]):(i + 1) * N] = \
            np.random.multivariate_normal(mean, cov / 2., size=N - len(gind[0]))
    return newX, newy


def KDE_resample(x, y, N, bandwidth=0.5):
    """
    Resample features based on Kernel Density approximation

    Parameters
    ----------
    X : numpy.ndarray
        Feature array
    y : numpy.ndarray
        Label array
    N : int
        Total samples to simulate (to be added to original sample)

    Returns
    -------
    newX : numpy.ndarray
        New Feature array
    newY : numpy.ndarray
        New label array
    """
    uys = np.unique(y)
    newX = np.zeros((int(N*len(uys)), np.size(x, axis=1)))
    newy = np.zeros((int(N*len(uys)), ))
    for i, uy in enumerate(uys):
        gind = np.where(y == uy)
        newX[i * N:i * N + len(gind[0]), :] = x[gind[0], :]
        newy[i * N:(i + 1) * N] = uy
        cx = x[gind[0], :]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(cx)
        newX[i * N + len(gind[0]):(i + 1) * N] = kde.sample(n_samples=N - len(gind[0]))
    return newX, newy


def prep_data_for_classifying(featurefile, means, stds, whiten=True, verbose=False):
    """
    Resample features based on Kernel Density approximation

    Parameters
    ----------
    featurefile : str
        File with pre-processed features
    means : numpy.ndarray
        Means of features, used to whiten
    stds : numpy.ndarray
        St. dev of features, used to whiten
    whiten : bool
        Whiten features before classification
    verbose : bool
        Print if SNe fail

    Returns
    -------
    X : numpy.ndarray
        Feature array
    final_sn_names : numpy.ndarray
        Label array
    means : numpy.ndarray
        Means of features, used to whiten
    stds : numpy.ndarray
        St. dev of features, used to whiten
    feat_names : numpy.ndarray
        Array of feature names
    """
    feat_data = np.load(featurefile, allow_pickle=True)
    ids = feat_data['ids']
    features = feat_data['features']
    feat_names = feat_data['feat_names']

    X = []
    final_sn_names = []
    for sn_name in ids:
        gind = np.where(sn_name == ids)
        if verbose:
            if len(gind[0]) == 0:
                print('SN not found')
                sys.exit(0)
            if not np.isfinite(features[gind][0]).all():
                print('Warning: ', sn_name, ' has a non-finite feature')
        if X == []:
            X = features[gind][0]
        else:
            X = np.vstack((X, features[gind][0]))
        final_sn_names.append(sn_name)

    gind = np.where(np.isnan(X))
    if len(gind) > 0:
        X[gind[0], gind[1]] = means[gind[1]]
    if whiten:
        X = (X - means) / stds

    return X, final_sn_names, means, stds, feat_names


def prep_data_for_training(featurefile, metatable, whiten=True):
    """
    Resample features based on Kernel Density approximation

    Parameters
    ----------
    featurefile : str
        File with pre-processed features
    metatable : numpy.ndarray
        Table which must include: Object Name, Redshift, Type, Estimate
        Peak Time, and EBV_MW
    whiten : bool
        Whiten features before classification

    Returns
    -------
    X : numpy.ndarray
        Feature array
    y : numpy.ndarray
        Label array
    final_sn_names : numpy.ndarray
        Label array
    means : numpy.ndarray
        Means of features, used to whiten
    stds : numpy.ndarray
        St. dev of features, used to whiten
    feat_names : numpy.ndarray
        Array of feature names
    """
    feat_data = np.load(featurefile, allow_pickle=True)
    ids = feat_data['ids']
    features = feat_data['features']
    feat_names = feat_data['feat_names']
    metadata = np.loadtxt(metatable, dtype=str, usecols=(0, 2))
    sn_dict = {'SLSN': 0, 'SNII': 1, 'SNIIn': 2, 'SNIa': 3, 'SNIbc': 4}

    X = []
    y = []
    final_sn_names = []
    for sn_name, sn_type in metadata:
        gind = np.where(sn_name == ids)
        if 'SN' not in sn_type:
            continue
        else:
            sn_num = sn_dict[sn_type]

        if len(gind[0]) == 0:
            continue
        if not np.isfinite(features[gind][0]).all():
            continue

        if X == []:
            X = features[gind][0]
            y = sn_num
        else:
            X = np.vstack((X, features[gind][0]))
            y = np.append(y, sn_num)
        final_sn_names.append(sn_name)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    if whiten:
        X = preprocessing.scale(X)

    return X, y, final_sn_names, means, stds, feat_names


def main():
    parser = ArgumentParser()
    parser.add_argument('metatable', type=str, default='', help='Get training set labels')
    parser.add_argument('--featurefile', default='./products/feat.npz', type=str, help='Feature file')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--train', action='store_true',
                        help='Train classification model')
    parser.add_argument('--savemodel', action='store_true', help='Save output model, training on full set')
    parser.add_argument('--add-random', dest='add_random', type=bool, default=False,
                        help='Add random number as feature (for testing)')
    parser.add_argument('--calc-importance', dest='calc_importance', type=bool,
                        default=False, help='Calculate feature importance')
    parser.add_argument('--only-raenn', dest='only_raenn', type=bool, default=False, help='Use ony RAENN features')
    parser.add_argument('--not-raenn', dest='not_raenn', type=bool, default=False, help='Exclude RAENN features')
    parser.add_argument('--no-int', dest='no_int', type=bool, default=False,
                        help='Exclude integral features (for testing)')
    parser.add_argument('--resampling', dest='resampling', type=str, default='KDE',
                        help='Resampling methods. Either KDE or Gauss available')
    parser.add_argument('--modelfile', dest='modelfile', type=str,
                        default='model', help='Name of model file to save')
    parser.add_argument('--randomseed', type=int, default=42, help='Name of model file to save')
    parser.add_argument('--outfile', dest='outfile', type=str,
                        default='superprob', help='Name of probability table file')
    args = parser.parse_args()

    sn_dict = {'SLSN': 0, 'SNII': 1, 'SNIIn': 2, 'SNIa': 3, 'SNIbc': 4}

    if args.train:
        X, y, names, means, stds, feature_names = prep_data_for_training(args.featurefile, args.metatable)
        names = np.asarray(names, dtype=str)
        if args.only_raenn:
            gind = [i for i, feat in enumerate(feature_names) if 'raenn' in feat]
            X = X[:, gind]
            feature_names = feature_names[gind]

        if args.not_raenn:
            gind = [i for i, feat in enumerate(feature_names) if 'raenn' not in feat]
            X = X[:, gind]
            feature_names = feature_names[gind]

        if args.no_int:
            gind = [i for i, feat in enumerate(feature_names) if 'int' not in feat]
            X = X[:, gind]
            feature_names = feature_names[gind]

        if args.add_random:
            feature_names = np.append(feature_names, 'random')

        if not args.savemodel:
            loo = LeaveOneOut()
            y_pred = np.zeros(len(y))
            for train_index, test_index in loo.split(X):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if args.resampling == 'Gauss':
                    X_res, y_res = Gauss_resample(X_train, y_train, 500)
                else:
                    X_res, y_res = KDE_resample(X_train, y_train, 500)

                new_ind = np.arange(len(y_res), dtype=int)
                np.random.shuffle(new_ind)
                X_res = X_res[new_ind]
                y_res = y_res[new_ind]

                if args.calc_importance:
                    X_res2, y_res2 = Gauss_resample(X_train, y_train, 500)
                    X_res2 = X_res2[:-40, :]
                    y_res2 = y_res2[:-40]

                if args.add_random:
                    X_res2, y_res2 = Gauss_resample(X_train, y_train, 500)
                    X_res2 = X_res2[:-40, :]
                    y_res2 = y_res2[:-40]
                    X_res = np.vstack((X_res.T, np.random.randn(len(X_res)))).T
                    X_res2 = np.vstack((X_res2.T, np.random.randn(len(X_res2)))).T
                    X_test = np.vstack((X_test.T, np.random.randn(len(X_test)))).T

                clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                             random_state=args.randomseed,
                                             criterion='gini',
                                             class_weight='balanced',
                                             max_features=None,
                                             oob_score=False)
                clf.fit(X_res, y_res)
                print(clf.predict_proba(X_test), y_test, names[test_index])

                if args.calc_importance:
                    feature_names = np.asarray(feature_names, dtype=str)
                    importances = clf.feature_importances_
                    indices = importances.argsort()[::-1]

                    print("Feature ranking:")

                    for f in range(X_res.shape[1]):
                        print(feature_names[indices[f]], importances[indices[f]])

                    plt.ylabel("Feature importances")
                    plt.bar(range(X_res.shape[1]), importances[indices],
                            color="grey", align="center")
                    plt.xticks(np.arange(len(importances))+0.5, feature_names[indices],
                               rotation=45, ha='right')
                    plt.show()
                y_pred[test_index] = np.argmax(clf.predict_proba(X_test))
            cnf_matrix = confusion_matrix(y, y_pred)
            print(cnf_matrix)
        if args.savemodel:
            if args.resampling == 'Gauss':
                X_res, y_res = Gauss_resample(X, y, 500)
            else:
                X_res, y_res = KDE_resample(X, y, 500)

            new_ind = np.arange(len(y_res), dtype=int)
            np.random.shuffle(new_ind)
            X_res = X_res[new_ind]
            y_res = y_res[new_ind]

            clf = RandomForestClassifier(n_estimators=350, max_depth=None,
                                         random_state=args.randomseed, criterion='gini', class_weight='balanced',
                                         max_features=None, oob_score=False)
            clf.fit(X_res, y_res)

            # save the model to disk
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
            if args.outdir[-1] != '/':
                args.outdir += '/'
            pickle.dump([clf, means, stds], open(args.outdir+args.modelfile+'_'+date+'.sav', 'wb'))
            pickle.dump([clf, means, stds], open(args.outdir+args.modelfile+'.sav', 'wb'))

    else:
        info = pickle.load(open(args.modelfile, 'rb'))
        loaded_model = info[0]
        means = info[1]
        stds = info[2]
        X, names, means, stds, feature_names = prep_data_for_classifying(args.featurefile, means, stds)
        names = np.asarray(names, dtype=str)
        probabilities = np.zeros((len(names), len(sn_dict)))
        for i, name in enumerate(names):
            probabilities[i] = loaded_model.predict_proba([X[i]])[0]
        probability_table = QTable(np.vstack((names, probabilities.T)).T,
                                   names=['Event Name', *sn_dict],
                                   meta={'name': 'SuperRAENN probabilities'})

        # save the model to disk
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if args.outdir[-1] != '/':
            args.outdir += '/'
        ascii.write(probability_table, args.outdir+args.outfile+'.tex',
                    format='latex', overwrite=True)


if __name__ == '__main__':
    main()

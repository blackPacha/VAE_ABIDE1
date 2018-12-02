#!/bin/python


'''
    Feature selection and classification models
    With numpy arrays
    Example: python analyze.py -dirname "./utils" -n 1 -target_path "./utils/Y.npy"
'''

import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import permutation_test_score
from sklearn.feature_selection import RFECV




def compute_mi(X1, X2):
    L_mi = []
    for i in range(X2.shape[1]):
        mi = mutual_info_classif(X1, X2[:, i])
        L_mi.append(mi)
    return np.array(L_mi)


def compute_reglog_l1_score(X_model, Y_model, X_val, Y_val):
    reglog = LogisticRegression(penalty='l1', solver='liblinear')
    reglog.fit(X_model, Y_model)
    L = (roc_auc_score(Y_model, reglog.predict_proba(X_model)[:, 1]),
        roc_auc_score(Y_val, reglog.predict_proba(X_val)[:, 1]),
        reglog.coef_)
    return L


def compute_reglog_l1_score_CV(X_model, Y_model, X_val, Y_val, skf_train):
    reglog = LogisticRegressionCV(cv=skf_train, scoring='roc_auc', penalty='l1', solver='liblinear')
    reglog.fit(X_model, Y_model)
    L = (roc_auc_score(Y_model, reglog.predict_proba(X_model)[:, 1]),
        roc_auc_score(Y_val, reglog.predict_proba(X_val)[:, 1]),
        reglog.coef_)
    return L


def compute_reglog_l2_score(X_model, Y_model, X_val, Y_val):
    reglog = LogisticRegression(penalty='l2')
    reglog.fit(X_model, Y_model)
    L = (roc_auc_score(Y_model, reglog.predict_proba(X_model)[:, 1]),
        roc_auc_score(Y_val, reglog.predict_proba(X_val)[:, 1]),
        reglog.coef_)
    return L


def compute_reglog_l2_score_CV(X_model, Y_model, X_val, Y_val, skf_train):
    reglog = LogisticRegressionCV(cv=skf_train, scoring='roc_auc', penalty='l2')
    reglog.fit(X_model, Y_model)
    L = (roc_auc_score(Y_model, reglog.predict_proba(X_model)[:, 1]),
        roc_auc_score(Y_val, reglog.predict_proba(X_val)[:, 1]),
        reglog.coef_)
    return L


def get_reg_log_score(X, y, penalty='l1'):
    skf_val = StratifiedKFold(n_splits=4, random_state=42)
    skf_train = StratifiedKFold(n_splits=3, random_state=42)
    splits = skf_val.split(X, y)
    # Build validation set and model building set
    for train_index, test_index in splits:
        X_model, X_val = X[train_index, :], X[test_index, :]
        Y_model, Y_val = y[train_index], y[test_index]
        break
    if penalty == 'l1':
        return compute_reglog_l1_score_CV(X_model, Y_model, X_val, Y_val, skf_train)
    else:
        return compute_reglog_l2_score_CV(X_model, Y_model, X_val, Y_val, skf_train)


def selectRFECV_RegLog(X, Y, print_plot=True, penalty = 'l2', solver='lbfgs', max_iter=100, n_splits=4, step = 1, verbose = 0, min_features_to_select=1):
    n_feat_before = X.shape[1]
    n_feat_after = X.shape[1] + 1
    # Create the RFE object and compute a cross-validated score.
    estimator = LogisticRegression(penalty = penalty, solver=solver, max_iter=max_iter)
    skf = StratifiedKFold(n_splits, random_state=42)
    rfecv = RFECV(estimator=estimator, step=step, cv=skf,
                  scoring='roc_auc', n_jobs=-1, verbose=verbose, min_features_to_select=min_features_to_select)
    rfecv.fit(X, Y)
    # Loop of selection
    while (n_feat_before != n_feat_after):
        n_feat_before = n_feat_after
        rfecv.fit(X, Y)
        X = X[:, rfecv.support_]
        n_feat_after = rfecv.n_features_
    # Option to get graph
    if print_plot:
        print("Optimal number of features : %d" % rfecv.n_features_)
        print(rfecv.support_)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        # Return selected features
        return rfecv.support_
    # Return only selected features - without graph
    else:
        return rfecv.support_


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Feature selection and classification of X / target')
    parser.add_argument('-dirname', type=str, required=True, help='directory name of encoded features in array')
    parser.add_argument('-n', nargs='+', required=True, help='list of model number')
    parser.add_argument('-target_path', type=str, required=True, help='Absolute path of the target.npy')
    args = parser.parse_args()

    y = np.load(args.target_path)
    #TODO: Test format and shape of y
    # preparation of data
    for n in args.n:

        dirname = args.dirname

        X = np.load(dirname + "/X_" + n + ".npy")
        # create dataframe to store scores
        df_score = pd.DataFrame()

        # first score with all the encoded features
        L_best = []
        skf = StratifiedKFold(n_splits=4, random_state=42)
        reglog = LogisticRegression('l2', solver='lbfgs')
        test = permutation_test_score(reglog,
                               X,
                               y,
                               cv=skf,
                               scoring='roc_auc')
        df_score['all_features'] = [test[0], test[2], X.shape[1]]
        L_best.append(list(range(X.shape[1])))

        skf_val = StratifiedKFold(n_splits=4, random_state=4)
        splits = skf_val.split(X, y)
        # Build test set and model building set
        for train_index, test_index in splits:
            X_model, X_val = X[train_index, :], X[test_index, :]
            Y_model, Y_val = y[train_index], y[test_index]
            break

        # Selection with Mutual Information between X and Y strictly positive
        mi = mutual_info_classif(X_model, Y_model)
        X_model_mi = X_model[:, list(map(lambda x: x > 0, list(mi)))]
        X_val_mi = X_val[:, list(map(lambda x: x > 0, list(mi)))]

        print("sel1:")
        sel1 = selectRFECV_RegLog(X_model_mi, Y_model, step=1, max_iter=1000, print_plot=False)
        print("\nsel2:")
        sel2 = selectRFECV_RegLog(X_model_mi, Y_model, step=2, max_iter=1000, print_plot=False)
        print("\nsel3:")
        sel3 = selectRFECV_RegLog(X_model_mi, Y_model, step=3, max_iter=1000, print_plot=False)
        print("\nsel4:")
        sel4 = selectRFECV_RegLog(X_model_mi, Y_model, step=5, max_iter=1000, print_plot=False)
        print("\nsel5:")
        sel5 = selectRFECV_RegLog(X_model_mi, Y_model, step=10, max_iter=1000, print_plot=False)
        print("\nsel6:")
        sel6 = selectRFECV_RegLog(X_model_mi, Y_model, step=15, max_iter=1000, print_plot=False)

        all_cols = pd.Series(range(X.shape[1]))
        sel1_rfe = [int(x in sel1) for x in all_cols]
        sel2_rfe = [int(x in sel2) for x in all_cols]
        sel3_rfe = [int(x in sel3) for x in all_cols]
        sel4_rfe = [int(x in sel4) for x in all_cols]
        sel5_rfe = [int(x in sel5) for x in all_cols]
        sel6_rfe = [int(x in sel6) for x in all_cols]

        df_sel = pd.DataFrame({"rfe_1": sel1_rfe,
                               "rfe_2": sel2_rfe,
                               "rfe_3": sel3_rfe,
                               "rfe_5": sel4_rfe,
                               "rfe_10": sel5_rfe,
                               "rfe_15": sel6_rfe})

        sel_common_vae_1 = list(all_cols[df_sel[df_sel.sum(1) >= 1].index])
        sel_common_vae_2 = list(all_cols[df_sel[df_sel.sum(1) >= 2].index])
        sel_common_vae_3 = list(all_cols[df_sel[df_sel.sum(1) >= 3].index])
        sel_common_vae_4 = list(all_cols[df_sel[df_sel.sum(1) >= 4].index])
        sel_common_vae_5 = list(all_cols[df_sel[df_sel.sum(1) >= 5].index])

        skf = StratifiedKFold(n_splits=3, random_state=42)
        reglog = LogisticRegression('l2', solver='lbfgs')

        if len(sel_common_vae_1) > 0:
            test = permutation_test_score(reglog,
                                          X[:, sel_common_vae_1],
                                          y,
                                          cv=skf,
                                          scoring='roc_auc')
            df_score["selected_features_1"] = [test[0], test[2], len(sel_common_vae_1)]
            L_best.append(sel_common_vae_1)
        if len(sel_common_vae_2) > 0:
            test = permutation_test_score(reglog,
                                          X[:, sel_common_vae_2],
                                          y,
                                          cv=skf,
                                          scoring='roc_auc')
            df_score["selected_features_2"] = [test[0], test[2], len(sel_common_vae_2)]
            L_best.append(sel_common_vae_2)
        if len(sel_common_vae_3) > 0:
            test = permutation_test_score(reglog,
                                          X[:, sel_common_vae_3],
                                          y,
                                          cv=skf,
                                          scoring='roc_auc')
            df_score["selected_features_3"] = [test[0], test[2], len(sel_common_vae_3)]
            L_best.append(sel_common_vae_3)
        if len(sel_common_vae_4) > 0:
            test = permutation_test_score(reglog,
                                          X[:, sel_common_vae_4],
                                          y,
                                          cv=skf,
                                          scoring='roc_auc')
            df_score["selected_features_4"] = [test[0], test[2], len(sel_common_vae_4)]
            L_best.append(sel_common_vae_4)
        if len(sel_common_vae_5) > 0:
            test = permutation_test_score(reglog,
                                          X[:, sel_common_vae_5],
                                          y,
                                          cv=skf,
                                          scoring='roc_auc')
            df_score["selected_features_5"] = [test[0], test[2], len(sel_common_vae_5)]
            L_best.append(sel_common_vae_5)

        # save df_score
        df_score.index = ["ROC_AUC", "p_val", "number_of_selected_features"]
        df_score.to_csv(dirname + "/df_score_" + n + ".csv")

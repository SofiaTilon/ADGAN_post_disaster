""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

def get_threshold(normal, abnormal):
    # Find natural cut of point between two distributions
    # https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves

    m1 = np.mean(normal)
    std1 = np.std(normal)
    m2 = np.mean(abnormal)
    std2 = np.std(abnormal)

    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)

    return np.roots([a, b, c])

def recall_(scores):
    # Compute Recall
    gt_labels = scores['gt_labels']
    an_scores = scores['an_scores']
    index_anomalies = np.where(gt_labels == 1)[0]
    index_normals = np.where(gt_labels == 0)[0]

    an_scores_normal = an_scores[index_normals]
    an_scores_abnormal = an_scores[index_anomalies]
    # find threshold
    result = get_threshold(an_scores_normal, an_scores_abnormal)
    threshold = np.max(result)

    bi_scores = np.copy(an_scores)
    bi_scores[bi_scores >= threshold] = 1
    bi_scores[bi_scores < threshold] = 0

    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, bi_scores)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return (recall, precision, accuracy, f1_score)

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap
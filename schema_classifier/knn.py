import pandas as pd
import gensim
import math
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, \
    multilabel_confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn import preprocessing
# from sklearn import metrics
import utils
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import interp
import pickle

# 0.8 Training - 0.2 Testing
PERCENT = 0.2

schemas = ["vulnerable", "angry", "impulsive", "happy", "detached", "punishing", "healthy"]
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]


def my_roc_curve(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(schemas)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(schemas))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(schemas)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(schemas)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    plt.figure()
    lw = 2

    # for i, color in zip(range(len(schemas)), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='{0} (AUC = {1:0.2f})'
    #                    ''.format(schemas[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue')

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('kNN ROC curves')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, roc_auc

def draw_roc(y_test, y_pred):
    fpr_knn, tpr_knn, roc_auc_knn = my_roc_curve(y_test, y_pred)

    # fpr_svm = dict()
    # tpr_svm = dict()
    # roc_auc_svm = dict()
    # fpr_rnn = dict()
    # tpr_rnn = dict()
    # roc_auc_rnn = dict()

    with open("../data/fpr_svm.pkl", "rb") as tf:
        fpr_svm = pickle.load(tf)

    with open("../data/tpr_svm.pkl", "rb") as tf:
        tpr_svm = pickle.load(tf)

    with open("../data/roc_auc_svm.pkl", "rb") as tf:
        roc_auc_svm = pickle.load(tf)

    with open("../data/fpr_rnn.pkl", "rb") as tf:
        fpr_rnn = pickle.load(tf)

    with open("../data/tpr_rnn.pkl", "rb") as tf:
        tpr_rnn = pickle.load(tf)

    with open("../data/roc_auc_rnn.pkl", "rb") as tf:
        roc_auc_rnn = pickle.load(tf)

    plt.figure()
    lw = 2

    plt.plot(fpr_knn["micro"], tpr_knn["micro"],
             label='kNN (AUC = {0:0.2f})'
                   ''.format(roc_auc_knn["micro"]),
             color='blue')

    plt.plot(fpr_svm["micro"], tpr_svm["micro"],
             label='SVM (AUC = {0:0.2f})'
                   ''.format(roc_auc_svm["micro"]),
             color='green')

    plt.plot(fpr_rnn["micro"], tpr_rnn["micro"],
             label='RNN (AUC = {0:0.2f})'
                   ''.format(roc_auc_rnn["micro"]),
             color='red')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle=':')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    lw = 2
    plt.plot(fpr_knn["macro"], tpr_knn["macro"],
             label='kNN (AUC = {0:0.2f})'
                   ''.format(roc_auc_knn["macro"]),
             color='blue')

    plt.plot(fpr_svm["macro"], tpr_svm["macro"],
             label='SVM (AUC = {0:0.2f})'
                   ''.format(roc_auc_svm["macro"]),
             color='green')

    plt.plot(fpr_rnn["macro"], tpr_rnn["macro"],
             label='RNN (AUC = {0:0.2f})'
                   ''.format(roc_auc_rnn["macro"]),
             color='red')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle=':')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average ROC curve')
    plt.legend(loc="lower right")
    plt.show()



def param_tuning_knn(model, labels, metric):
    X = model.docvecs.vectors_docs
    y = labels

    k_range = list(range(4, 60))
    weight_options = ["uniform", "distance"]
    distance_metrics = [1, 2]  # 1=Manhattan, 2=Euclidean
    param_grid = dict(estimator__n_neighbors=k_range, estimator__weights=weight_options, estimator__p=distance_metrics)

    knn = KNeighborsClassifier()
    clf = MultiOutputClassifier(knn, n_jobs=-1)

    grid = GridSearchCV(clf, param_grid, cv=10, scoring=metric, n_jobs=-1)
    grid.fit(X, y)

    print(metric, grid.best_score_, grid.best_params_)
    # print(grid.best_estimator_)


def spearman(y_pred, y_test):
    gof_spear = np.zeros(y_test.shape[1])
    for schema in range(len(schemas)):
        rho, p = stats.spearmanr(y_pred[:, schema], y_test[:, schema])
        gof_spear[schema] = rho

    # performance per schema
    print(pd.DataFrame(data=gof_spear, index=schemas, columns=['gof']))

    # mean performance
    mean = np.nanmean(gof_spear, dtype=np.float64)
    print("mean: " + str(mean))

    return mean


def find_k(model, labels, reg=False):
    best_k = 0
    best_mean = 0
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 100]:
        y_test, y_pred = my_knn(model, labels, k, reg)
        print('K: ' + str(k))
        mean = spearman(y_pred, y_test)
        if best_mean < mean:
            best_mean = mean
            best_k = k

    print(str(best_k) + " " + str(best_mean))
    return best_k, best_mean


def param_tuning_mlknn(model, labels, metric):
    X = model.docvecs.vectors_docs
    y = labels

    k_range = list(range(2, 60))
    smoothing_params = [1.0]

    param_grid = dict(k=k_range, s=smoothing_params)
    mlknn = MLkNN()

    grid = GridSearchCV(mlknn, param_grid, scoring=metric, n_jobs=-1)
    grid.fit(X, y)

    print(metric, grid.best_score_, grid.best_params_)


def metric_tuning(model, labels):
    # Overview of scoring parameters
    # https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    score_metrics = ["accuracy", "f1_weighted", "f1_samples"]
    for i, metric in enumerate(score_metrics):
        param_tuning_knn(model, labels, metric)


def report(y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=schemas))


def my_mlknn(model: gensim.models.doc2vec.Doc2Vec, labels):
    X = model.docvecs.vectors_docs
    np_x = np.asarray(X)
    np_y = np.asarray(labels)
    x_train, y_train, x_test, y_test, percent = utils.split_data(np_x, np_y, PERCENT)

    mlknn = MLkNN(k=40)

    mlknn.fit(x_train, y_train)
    print("FITTED")

    y_pred = mlknn.predict(x_test)
    print("PREDICTION")
    print("K=" + str(mlknn.k) + ", accuracy score: " + str(accuracy_score(y_test, y_pred)))


def my_knn(model: gensim.models.doc2vec.Doc2Vec, labels, k, reg=False):
    X = model.docvecs.vectors_docs
    np_x = np.asarray(X)
    np_y = np.asarray(labels)
    x_train, y_train, x_test, y_test, percent = utils.split_data(np_x, np_y, PERCENT)
    y_preds = 0
    if (reg):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_preds = knn.predict(x_test)
    else:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf = MultiOutputClassifier(knn, n_jobs=-1)
        clf = clf.fit(x_train, y_train)
        y_preds = clf.predict(x_test)


    return y_test, y_preds


# PLOTS
def k_means_scatter_plot(model):
    X = model.docvecs.vectors_docs

    # Perform Dimensionality Reduction
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    # K-means for visualization of possbile clusters
    kmeans = KMeans(n_clusters=7, max_iter=100)
    kmeans.fit(principal_components)
    y_kmeans = kmeans.predict(principal_components)

    # Scatter plot
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y_kmeans, s=15, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    plt.title("Doc2Vec vectorspace (Unlabelled samples + PCA)")
    plt.show()


def get_knn_accuracy(model, labels):
    accuracy = []

    for i, schema in enumerate(schemas):
        X = model.docvecs.vectors_docs
        np_x = np.asarray(X)
        np_y = np.asarray(labels[:, i])

        x_train, y_train, x_test, y_test, percent = utils.split_data(np_x, np_y, 0.2)

        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        accuracy.append(round(accuracy_score(y_test, y_pred), 2))
    return accuracy


def print_accuracy(accuracy):
    for i, schema in enumerate(schemas):
        print(schema + ": " + str(accuracy[i]))

    avg_accuracy = 0
    for acc in accuracy:
        avg_accuracy += acc

    avg_accuracy = avg_accuracy / len(accuracy)
    print("Mean_accuracy" + ": " + str(round(avg_accuracy, 2)))


def knn_accuracy_table(model, labels):
    data = get_knn_accuracy(model, labels)
    for i, acc in enumerate(data):
        plt.bar(i, acc, color=colors[i])

    plt.xlabel("Schemas")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks([i for i, _ in enumerate(schemas)], schemas)
    plt.title("Binary classification with KNN")
    plt.show()


def get_confusion_matrix(y_test, y_pred):
    matrices = multilabel_confusion_matrix(y_test, y_pred)
    i = 0
    for mat in matrices:
        cmd = ConfusionMatrixDisplay(mat, display_labels=np.unique(y_test)).plot()
        accuracy = str(round(accuracy_score(y_test[:, i], y_pred[:, i]), 2))
        plt.title(schemas[i] + " (Accuracy: " + accuracy + ")")
        plt.xlabel("Actual label")
        plt.show()
        i += 1


if __name__ == '__main__':
    df = pd.read_csv("../data/FINAL_CSV.csv")

    texts, bin_labels = utils.get_text_labels(df)
    # texts, ordinal_labels = utils.get_average_for_each_label(df)
    # proc_text, tokenized_texts = utils.pre_process_data(texts)


    # training d2v model
    # utils.training_model_d2v(tokenized_texts)

    print("LOADING MODEL")
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('../models/schema-d2v-knn.model')

    # y_test, y_preds = my_knn(d2v_model, labels)

    y_test, y_pred = my_knn(d2v_model, bin_labels, 4)

    # print(y_test[:, 1])
    # my_roc_curve(y_test, y_pred)
    draw_roc(y_test, y_pred)

    # get_confusion_matrix(y_test, y_pred)
    # report(y_test, y_pred)

    # param_tuning_knn(d2v_model, labels, 'f1_weighted')

    # print("PARAM TUNING")
    # param_tuning_knn(d2v_model, bin_labels, "f1_weighted")

    # print("GOF Spearman:")
    # print("KNN classification")
    # find_k(d2v_model, ordinal_labels)
    # print("KNN Regression:")
    # find_k(d2v_model, ordinal_labels, True)
    # y_test, y_pred = my_knn(d2v_model, ordinal_labels, 1)
    # print(str(accuracy_score(y_test, y_pred)))
    # spearman(y_pred, y_test)

    # # scaling to min_max range
    # min_max_scaler = preprocessing.MinMaxScaler()
    # y_test = min_max_scaler.fit_transform(y_test)

    # print("------------")
    # report(y_test, y_preds)
    # print(y_pred)
    # Report

    # report(y_test, y_pred)
    print("CHECK")


from sklearn.metrics import *

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt


def evaluate_model(model,X_test,y_test,categories):

    y_pred=model.predict(X_test)

    y_pred=np.argmax(y_pred,axis=1)

    y_true=np.argmax(y_test,axis=1)

    cm=confusion_matrix(y_true,y_pred)

    print(classification_report(
        y_true,
        y_pred,
        target_names=categories
    ))

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=categories,
        yticklabels=categories
    )

    plt.show()

    return cm


def calculate_metrics(cm):

    sensitivity=[]

    specificity=[]

    total=np.sum(cm)

    for i in range(len(cm)):

        TP=cm[i][i]

        FN=np.sum(cm[i,:])-TP

        FP=np.sum(cm[:,i])-TP

        TN=total-(TP+FN+FP)

        sensitivity.append(
            TP/(TP+FN)
        )

        specificity.append(
            TN/(TN+FP)
        )

    return sensitivity,specificity
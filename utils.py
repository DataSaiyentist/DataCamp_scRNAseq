import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc


# Pairplot of PCA components
def plot_acp(X, y, num_components=25):
    """Plot pairwise relationships betwwen all PCA components"""

    X_pca = PCA(n_components=num_components).fit_transform(X)

    # Create a dataframe with the PCA components
    df_pca = pd.DataFrame(X_pca)
    df_pca["cell_type"] = y

    # Plot the pairwise relationships
    sns.pairplot(df_pca, hue="cell_type")
    plt.show()


# t-SNE (2D visualization of the dataset)
def plot_tsne(X, y):
    """Plot the t-SNE of the dataset"""

    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)

    # Create a dataframe with the t-SNE data
    df_tsne = pd.DataFrame(
        {
            "1st_component": X_tsne[:, 0],
            "2nd_component": X_tsne[:, 1],
            "cell_type": y,
        }  # noqa
    )

    # Plot the t-SNE
    sns.scatterplot(
        data=df_tsne,
        x="1st_component",
        y="2nd_component",
        hue="cell_type",
        palette="bright",
    )
    plt.show()


# Genes expresssion for each cell
def plot_cell(X, per_plot=10):
    """Plot genes expression for each cell"""

    sections = np.array_split(X, np.ceil(len(X) / per_plot))

    for i, section in enumerate(sections):
        plt.plot(
            section.T,
            label=[f"Lignes {i*per_plot+j}" for j in range(section.shape[0])],  # noqa
        )
        plt.xlabel("Index of genes")
        plt.ylabel("Values")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8))
        plt.show()


# Expression of a gene for each cell-type
def plot_gene(X, y):
    """Plot gene expressions for each cell-type"""

    df = pd.DataFrame(X, columns=[f"gene_{i + 1}" for i in range(X.shape[1])])
    df["cell_type"] = y

    for column in df.columns[:-1]:
        sns.violinplot(x="cell_type", y=column, data=df, palette="Set2")
        plt.title(column)
        plt.show()


# Call-rate of each cell
def plot_cr_cell(X, threshold=0.1):
    """Plot cell call-rates"""

    plt.plot(X.mean(axis=1))  # Call-rate of cells
    plt.axhline(y=threshold, color="r", linestyle="-", label="Threshold")
    plt.xlabel("Index of cells")
    plt.ylabel("Call-rates")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.show()


# Call-rate of each gene
def plot_cr_gene(X, threshold=0.1):
    """Plot gene call-rates"""

    plt.plot(X.mean(axis=0))  # Call-rate of genes
    plt.axhline(y=threshold, color="r", linestyle="-", label="Threshold")
    plt.xlabel("Index of genes")
    plt.ylabel("Call-rates")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.show()


# Variance of each gene
def plot_var_gene(X, threshold=0.1):
    """Plot gene variances"""

    plt.plot(X.var(axis=0))  # Variance of genes expression
    plt.axhline(y=threshold, color="r", linestyle="-", label="Threshold")
    plt.xlabel("Index of genes")
    plt.ylabel("Variances")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.show()


# Hyperparametrization automation
def search_param(
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    param_grid,
    cv=10,
    is_balanced=False,  # noqa
):
    """Give the best parameters for a given ML model"""

    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy" if is_balanced else "balanced_accuracy",
    )
    clf.fit(X_train, y_train)

    print(clf.best_params_)  # Best parameters
    # Accuracy of the model with the best parameters
    if is_balanced:
        print(f"Train: {accuracy_score(y_train, clf.predict(X_train))}")
        print(f"Test: {accuracy_score(y_test, clf.predict(X_test))}")
    else:
        print(
            f"Train: {balanced_accuracy_score(y_train, clf.predict(X_train))}"
        )  # noqa
        print(f"Test: {balanced_accuracy_score(y_test, clf.predict(X_test))}")


# ROC curve of a model
def plot_roc(X, y, model):
    """Plot the ROC curve of a given ML model"""

    n_classes = len(model.classes_)
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            y == model.classes_[i], model.predict_proba(X)[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"AUC = {roc_auc[i]:.2f} for {model.classes_[i]}",  # noqa
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.legend(loc="lower right")
    plt.show()


# Model evaluation
def evaluate_model(
    X_train, X_test, y_train, y_test, model, name, is_balanced=False
):  # noqa
    """Evaluate some metrics of a given ML model"""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = (
        accuracy_score(y_test, y_pred)
        if is_balanced
        else balanced_accuracy_score(y_test, y_pred)
    )
    print(f"{name} Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )
    disp.plot()
    plt.title(f"{name} Confusion matrix")
    plt.figure(figsize=(7, 4))

    # ROC curve
    plot_roc(X_test, y_test, model)
    # Visualize predicted labels with t-SNE
    plot_tsne(X_test, y_pred)


# Cross-validation of all considered models
def validate_model(X, y, models, kfold=10, is_balanced=False):
    """Cross-validate ML models"""

    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

    # Accuracy of models
    train_scores, test_scores = [], []
    metric = accuracy_score if is_balanced else balanced_accuracy_score
    for name, model in models.items():
        # (Stratified) K-fold cross-validation for each model
        train_scores_cv, test_scores_cv = [], []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            train_scores_cv.append(metric(y_train, model.predict(X_train)))
            test_scores_cv.append(metric(y_test, model.predict(X_test)))

        train_scores.append(train_scores_cv)
        test_scores.append(test_scores_cv)

    # Plot for each model
    fig, axs = plt.subplots(nrows=len(models), ncols=2, figsize=(10, 17))
    for i, (name, model) in enumerate(models.items()):
        sns.violinplot(data=train_scores[i], ax=axs[i][0], color="orange")
        axs[i][0].set_title(f"{name} train scores")
        axs[i][0].xaxis.set_visible(False)
        sns.violinplot(data=test_scores[i], ax=axs[i][1], color="forestgreen")
        axs[i][1].set_title(f"{name} test scores")
        axs[i][1].xaxis.set_visible(False)

    plt.show()

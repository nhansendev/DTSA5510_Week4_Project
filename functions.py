import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from itertools import product
from copy import deepcopy
import time
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression


def generate_PCA(dataset_df, categories, random_state=0):
    pca_train_sets = []
    pca_test_sets = []
    pca_times = []
    cnums = []
    for i in range(len(dataset_df)):
        # For each dataset
        st = time.time()
        # Fit a full PCA model
        pca = PCA(random_state=random_state).fit(dataset_df['Train'][i])
        ex_var_csum = np.cumsum(pca.explained_variance_ratio_)
        # Identify the number of components to reach 95% explained variance
        N = np.argmax(ex_var_csum>0.95)
        cnums.append(N)
        # Fit another model with reduced components count
        pca = PCA(N, random_state=random_state).fit(dataset_df['Train'][i])
        pca_times.append(time.time()-st)
        
        # Create transformed datasets
        tfm = pca.transform(dataset_df['Train'][i])
        print(f'{i}: {dataset_df['Train'][i].shape} -> {tfm.shape}')

        pca_data = pd.DataFrame(tfm)
        pca_data['cats'] = categories
        pca_train_sets.append(pca_data)
        
        pca_test_sets.append(pca.transform(dataset_df['Test'][i]))

    dataset_df['PCA_Comp'] = cnums
    dataset_df['PCA_Train'] = pca_train_sets
    dataset_df['PCA_Test'] = pca_test_sets
    dataset_df['PCA_Time'] = pca_times


def plot_performance(accs, acc_labels, times, time_labels, dataset_df, size=(13, 4), tlow=0.1, thigh=1000):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(size)
    axs = fig.axes

    X = range(len(dataset_df))

    for a in accs:
        axs[0].plot(X, a, '.-')
    axs[0].grid()
    axs[0].legend(acc_labels)
    axs[0].set_xticks(X)
    axs[0].set_xlabel('Dataset')
    axs[0].set_ylabel('Test Accuracy')

    for t in times:
        axs[1].plot(X, t, '.-')
    axs[1].set_ylim(tlow, thigh)
    axs[1].legend(time_labels)
    axs[1].set_xticks(X)
    axs[1].set_xlabel('Dataset')
    axs[1].set_ylabel('Training Time (s)')
    axs[1].set_yscale('log')
    axs[1].grid()

    rows = []
    cols = ['DocLower', 'DocUpper', 'Ngrams', 'Features', 'PCA_Comp']
    for i in range(len(X)):
        rows.append([str(dataset_df[c][i]) for c in cols])
    rcols = ['0.7']*len(rows)
    ccols = ['0.7']*len(cols)
    tbl = axs[2].table(rows, rowLabels=X, colLabels=cols, loc='center', rowColours=rcols, colColours=ccols)
    tbl.auto_set_column_width(range(len(cols)))
    for (row, col), cell in tbl.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    axs[2].axis('off')
    plt.tight_layout()
    plt.show()


class BaseGridSearch:
    def data_to_XY(self, data, target=None):
        if target is None:
            target = self.target_col
        # Separate dataframe into target (Y) and everything else (X)
        return data.drop(columns=[target]), data[target]

    def kfold_split(self, data):
        # Split data for K-fold cross-validation
        KF = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_seed)
        train = []
        test = []
        for train_idx, test_idx in KF.split(data):
            train.append(self.data_to_XY(data.iloc[train_idx]))
            test.append(self.data_to_XY(data.iloc[test_idx]))
        return train, test

    def simple_split(self, data, test_frac=0.2):
        # Generic test-train split
        data_train, data_test = train_test_split(
            data, test_size=test_frac, random_state=self.random_seed
        )
        return self.data_to_XY(data_train), self.data_to_XY(data_test)


class NMF_GridSearch(BaseGridSearch):
    def __init__(self, dataframe, n_components, params, max_iter=1500, target_col='cats', cv=5, random_seed=0):
        self.data = dataframe
        self.n_components = n_components
        self.params = params
        self.max_iter = max_iter
        self.cv = cv
        self.target_col = target_col
        self.random_seed = random_seed
        self.models = []
        self.scores = []
        self.times = []

        self.make_paramsets()

        train, test = self.simple_split(self.data)
        self.train_x, self.train_y = train
        self.test_x, self.test_y = test

        self.train_data, self.test_data = self.kfold_split(dataframe)

    def make_paramsets(self):
        keys = list(self.params.keys())
        if len(keys) > 1:
            vals = list(product(*list(self.params.values())))
        else:
            vals = [[v] for v in self.params[keys[0]]]

        self.paramsets = []
        for i in range(len(vals)):
            paramset = {}
            for j, k in enumerate(keys):
                paramset[k] = vals[i][j]
            self.paramsets.append(paramset)
    
    def fit(self, verbose=True, use_mode=True):
        N = len(self.paramsets)
        for i, paramset in enumerate(self.paramsets):
            if verbose:
                print(f"Evaluating Parameter Set {i+1} of {N}: {paramset}")
        
            model = NMF_instance(self.n_components, paramset, max_iter=self.max_iter, random_state=self.random_seed)
            
            st = time.time()
            train_score = 0
            test_score = 0
            try:
                for i in range(self.cv):
                    score, dt = model.fit_eval(*self.train_data[i], use_mode=use_mode)
                    train_score += score / self.cv

                    score = model.eval_accuracy(*self.test_data[i])
                    test_score += score / self.cv
            
            except ValueError:
                if verbose:
                    print('Skipping: parameter combination not allowed')
                self.models.append(None)
                self.scores.append([0, 0])
                continue
            dt = time.time()-st

            if verbose:
                print(f'Train: {train_score:.3f}, Test: {test_score:.3f}, Time: {dt:.3f} sec')
            self.models.append(model)
            self.scores.append([train_score, test_score])
        
        return self.get_best_model()
            
    def get_best_model(self):
        idx = np.argmax([s[1] for s in self.scores])
        model = self.models[idx]
        train_score, dt = model.fit_eval(self.train_x, self.train_y)
        test_score = model.eval_accuracy(self.test_x, self.test_y)
        return model, train_score, test_score, dt
    

class NMF_instance:
    def __init__(self, n_components, params, max_iter=1500, random_state=0):
        self.model = NMF(n_components, **params, max_iter=max_iter, random_state=random_state)
        self.params = params
        self.catmap = None
        self.invcatmap = None
        self.acc = None

    def fit_eval(self, X, categories, use_mode=True):
        st = time.time()
        self.model.fit(X)
        dt = time.time()-st
        self.get_catmap(X, categories, use_mode)
        return self.eval_accuracy(X, categories), dt

    def get_catmap(self, X, categories, use_mode=True):
        cats = np.unique(categories)
        values = []
        if use_mode:
            for c in cats:
                preds = self.predict(X[categories == c])
                vals, counts = np.unique_counts(preds)
                I = np.argmax(counts)
                values.append(int(vals[I]))
            pairs = list(zip(values, cats))
            pairs.sort()
            self.catmap = {k: v for v, k in pairs}
        else:
            for c in cats:
                pred = np.mean(self.predict(X[categories == c]))
                values.append(pred)            
            pairs = list(zip(values, cats))
            pairs.sort()
            self.catmap = {k: i for i, k in enumerate([p[1] for p in pairs])}
        return self.catmap
    
    def get_invcatmap(self):
        self.invcatmap = {v: k for k, v in self.catmap.items()}
        return self.invcatmap
        
    def predict(self, X):
        return np.argmax(self.model.transform(X), 1)
    
    def eval_accuracy(self, X, categories):
        if isinstance(categories, pd.DataFrame) or isinstance(categories, pd.Series):
            mapped = categories.map(self.catmap)
        else:
            mapped = []
            for r in range(len(categories)):
                mapped.append(self.catmap[categories[r]])
        self.acc = accuracy_score(mapped, self.predict(X))
        return self.acc
        

def train_NMF(dataset_df, categories, params, random_state=0, verbose=False):
    N = len(dataset_df)
    cats = categories.unique()
    accs = []
    models = []
    times = []
    invcatmaps = []
    for i in range(N):
        if verbose:
            print(f'Training {i+1} of {N}')
        row = dataset_df.iloc[i]
        st = time.time()
        model = NMF(n_components=5, **params, random_state=random_state).fit(row['Train'])
        models.append(model)
        W = model.transform(row['Train'])
        times.append(time.time()-st)

        # Determine mapping between categories and transformed labels
        values = []
        for c in cats:
            idx = categories == c
            preds = np.argmax(model.transform(row['Train'][idx]), 1)
            vals, counts = np.unique_counts(preds)
            I = np.argmax(counts)
            values.append(int(vals[I]))
        pairs = list(zip(values, cats))
        pairs.sort()
        catmap = {k: v for v, k in pairs}
        icm = {v: k for v, k in pairs}
        
        train_acc = accuracy_score(categories.map(catmap), np.argmax(W, 1))
        accs.append(train_acc)
        invcatmaps.append(icm)

    dataset_df['TrainAcc'] = accs
    dataset_df['Time_NMF'] = times
    dataset_df['ICM_NMF'] = invcatmaps
    dataset_df['Model_NMF'] = models


def train_LR(dataset_df, categories, params, random_state=0, verbose=False):
    N = len(dataset_df)
    accs = []
    models = []
    times = []
    for i in range(N):
        if verbose:
            print(f'Training {i+1} of {N*2}')
        row = dataset_df.iloc[i]
        st = time.time()
        model = LogisticRegression(**params, random_state=random_state).fit(row['Train'], categories)
        models.append(model)
        times.append(time.time()-st)
        
        train_acc = accuracy_score(categories, model.predict(row['Train']))
        accs.append(train_acc)
    
    model_df = pd.DataFrame({'TrainAcc': accs, 'Time_LR': times, 'model_LR': models})

    accs = []
    models = []
    times = []
    for i in range(N):
        if verbose:
            print(f'Training {i+1+N} of {N*2}')
        row = dataset_df.iloc[i]
        st = time.time()
        dset = row['PCA_Train'].drop(columns=['cats'])
        model = LogisticRegression(**params, random_state=random_state).fit(dset, categories)
        models.append(model)
        times.append(time.time()-st)
        
        train_acc = accuracy_score(categories, model.predict(dset))
        accs.append(train_acc)

    model_df['PCA_TrainAcc_LR'] = accs
    model_df['PCA_Time_LR'] = times #dataset_df['PCA_Time']
    model_df['PCA_Model_LR'] = models

    return model_df


def plot_occ(words_df, doc_levels=[5, 20, 200, 450, 750], occ_levels=[10, 100, 1000, 10000], size=(12, 4)):
    fig, ax = plt.subplots(1, 3)
    axs = fig.axes
    fig.set_size_inches(size)

    W = max(words_df['Occ'])

    for a in axs[:2]:
        a.plot(words_df['Occ'], words_df['DocOcc'], '.')
        a.hlines(doc_levels, 0, W, linestyles='dashed', color='k')
        for L in doc_levels:
            a.text(W//2, L, f'{L}', va='bottom', ha='center')
        a.set_xscale('log')
        a.set_xlabel('Word Occurrences')
        a.set_ylabel('Document Occurrences')
        a.grid()

    axs[1].set_yscale('log')

    axs[2].plot(words_df['Occ'])
    axs[2].hlines([words_df['Occ'][v] for v in occ_levels], 0, len(words_df['Occ']), color='k', linestyles='dashed')
    for v in occ_levels:
        axs[2].text(len(words_df['Occ'])//2, words_df['Occ'][v], f'↑ Top {v} (>{words_df['Occ'][v]} occ.)', ha='center', va='bottom')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Word Index')
    axs[2].set_ylabel('Word Occurrences')

    plt.tight_layout()
    plt.show()


def pred_to_csv(pred, ids, basedir, inv_catmap=None, name="submission"):
    df = pd.DataFrame(columns=["ArticleId", "Category"])
    df["ArticleId"] = ids
    df["Category"] = pred
    if inv_catmap is not None:
        df["Category"] = df["Category"].map(inv_catmap)

    fpath = os.path.join(basedir, f"{name}.csv")
    df.to_csv(fpath, index=False)
    return fpath


def try_format_float(value, fmt=".1e"):
    if isinstance(value, bool):
        return value
    try:
        return f"{value:{fmt}}"
    except ValueError:
        return value


def plot_manual_search_grid(train_scores, test_scores, params, size=(16, 8), numformat='.1e'):
    # Visualize the results of performing grid search
    keys = list(params.keys())
    vals = list(params.values())

    fig, ax = plt.subplots(1, 2)
    axs = fig.axes
    fig.set_size_inches(size)

    axs[0].imshow(train_scores, interpolation="nearest", cmap="RdYlGn")
    axs[1].imshow(test_scores, interpolation="nearest", cmap="RdYlGn")

    # Add numeric labels to supplement the colors in the grid
    for i in range(train_scores.shape[0]):
        for j in range(train_scores.shape[1]):
            axs[0].text(j, i, f"{train_scores[i][j]:.3f}", ha="center", va="center")
            axs[1].text(j, i, f"{test_scores[i][j]:.3f}", ha="center", va="center")

    axs[0].set_ylabel(keys[0])
    axs[0].set_xlabel(keys[1])
    axs[1].set_ylabel(keys[0])
    axs[1].set_xlabel(keys[1])

    xfmt = [try_format_float(v, numformat) for v in vals[1]]
    yfmt = [try_format_float(v, numformat) for v in vals[0]]
    axs[0].set_xticks(np.arange(len(vals[1])), xfmt, rotation=45)
    axs[0].set_yticks(np.arange(len(vals[0])), yfmt)
    axs[1].set_xticks(np.arange(len(vals[1])), xfmt, rotation=45)
    axs[1].set_yticks(np.arange(len(vals[0])), yfmt)
    axs[0].set_title("Training F1-Scores")
    axs[1].set_title("Testing F1-Scores")
    plt.tight_layout()
    plt.show()


class GridSearch(BaseGridSearch):
    # A custom implementation of grid search, including cross-validation and plotting
    def __init__(
        self, data, params, estimator, target_col="cats", cv=3, random_seed=0
    ) -> None:
        self.params = params
        self.estimator = estimator
        self.cv = cv  # cross-validation qty
        self.data = data
        self.random_seed = random_seed
        self.target_col = target_col

        train, test = self.simple_split(self.data)
        self.train_x, self.train_y = train
        self.test_x, self.test_y = test
        self.train_data, self.test_data = self.kfold_split(data)

        self.best_test_model = None
        self.train_score = None
        self.test_score = None

        self.keys = list(params.keys())
        self.val_lens = [len(v) for v in params.values()]
        vals = list(product(*list(params.values())))
        self.row_map = {str(p): i for i, p in enumerate(vals)}
        vals = [list(v) + [0.0, 0.0] for v in vals]
        cols = self.keys + ["train", "test"]
        self.data_df = pd.DataFrame(vals, columns=cols)

    def get_scores(self, verbose=True):
        # Calculate training accuracy as well as
        # testing accuracy, recall, and precision
        if self.best_test_model is None:
            self.get_best_test_model()

        score = self.test(self.best_test_model, self.train_x, self.train_y)
        self.train_score = score
        if verbose:
            print(f"Train Accuracy: {score:.3f}")

        pred = self.best_test_model.predict(self.test_x)
        self.test_score = accuracy_score(self.test_y, pred)
        if verbose:
            print(f"Test Accuracy: {self.test_score:.3f}")
        return {
            "train": self.train_score,
            "test": self.test_score,
            "precision": precision_score(self.test_y, pred, average="micro"),
            "recall": recall_score(self.test_y, pred, average="micro"),
        }

    def plot_confusion_matrix(self, size=(4, 4), xrot=45):
        # Show the test confusion matrix of the best model
        if self.best_test_model is None:
            self.get_best_test_model()

        pred = self.best_test_model.predict(self.test_x)
        labels = self.test_y.unique()
        conf_mat = confusion_matrix(self.test_y, pred, labels=labels)
        cm_plot = ConfusionMatrixDisplay(conf_mat, display_labels=labels)
        cm_plot.plot()
        plt.gcf().set_size_inches(size)
        plt.title("Training Confusion Matrix")
        ax = plt.gca()
        T = ax.get_xticks()
        TL = ax.get_xticklabels()
        ax.set_xticks(T, TL, rotation=xrot)
        plt.show()

    def plot_search_grid(self, size=(16, 8), numformat='.1e'):
        # Show the result of the parameter grid search
        plot_manual_search_grid(*self.get_matrices(), self.params, size=size, numformat=numformat)

    def copy_estimator(self):
        return deepcopy(self.estimator)

    def get_best_test_model(self):
        # Train an instance of the estimator using the best parameter set
        model = self.copy_estimator()
        params = self.get_best_test_params()
        for k, v in params.items():
            model.__dict__[k] = v
        st = time.time()
        model.fit(self.train_x, self.train_y)
        dur = time.time()-st
        self.best_test_model = model
        return model, dur

    def fit(self, verbose=True):
        # Perform grid search
        st = time.time()
        dlen = len(self.data_df)
        for row in range(len(self.data_df)):
            if verbose and not row % (dlen // 10):
                print(f"Evaluating Parameter Set: {row+1}/{dlen} ({(row+1)/dlen:.1%})")
            est = self.copy_estimator()
            for k in self.keys:
                est.__dict__[k] = self.data_df.loc[row, k]
            st = time.time()
            for i in range(self.cv):
                X, Y = self.train_data[i]
                est.fit(X, Y)
                score = self.test(est, *self.train_data[i])
                self.data_df.at[row, "train"] += score / self.cv

                score = self.test(est, *self.test_data[i])
                self.data_df.at[row, "test"] += score / self.cv
            if verbose:
                dt = time.time()-st
                print(f'Train: {self.data_df.at[row, "train"]:.3f}, Test: {self.data_df.at[row, "test"]:.3f}, Time: {dt:.3f} sec')
        if verbose:
            print("Done")
        return time.time() - st

    def eval(self, scores, times, models, params, plot_confusion=True, verbose=True):
        # Perform grid search and report performance, with optional plots
        duration = self.fit(verbose)
        stats = self.get_scores(verbose)
        scores.append(stats['test'])
        # times.append(duration)
        models.append(self)
        params.append(self.get_best_test_params())
        if verbose:
            print("Best Parameters: ", self.get_best_test_params())
        if plot_confusion:
            self.plot_confusion_matrix()

    def test(self, model, X, Y):
        # Evaluate a given model's performance
        return accuracy_score(Y, model.predict(X))

    def get_matrices(self):
        # Get testing and training scores as numpy matrices (for grid plot)
        train_scores = np.array(self.data_df["train"]).reshape(*self.val_lens)
        test_scores = np.array(self.data_df["test"]).reshape(*self.val_lens)
        return train_scores, test_scores

    def get_best_test_params(self):
        row = self.data_df["test"].idxmax()
        return {k: self.data_df.loc[row, k] for k in self.keys}


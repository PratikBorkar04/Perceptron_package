
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
def prepare_data(df):
  X = df.drop("y",axis = 1)
  y = df["y"]
  return X,y

def save_model(model,filename):
  model_dir = "models"
  os.makedirs(model_dir,exist_ok=True)
  filepath = os.path.join(model_dir,filename)
  joblib.dump(model,filepath)

def save_plot(df, file_name, model):
    """
    :param df: its a dataframe
    :param file_name: its path to save the plot
    :param model: trained model
    """
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        colors = ("red", "blue", "lightgreen", "gray", "cyan")
        colormap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values  # as a array
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=colormap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()
    X, y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    plotPath = os.path.join(plot_dir, file_name)  # model/filename
    plt.savefig(plotPath)

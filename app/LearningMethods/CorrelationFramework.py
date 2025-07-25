import re
import pandas as pd
from LearningMethods.correlation_evaluation import SignificantCorrelation
import Plot.plot_positive_negative_bars as PP
import Plot.plot_real_and_shuffled_hist as PPR
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt

from Plot.New_taxtree_draw import draw_tree

from Plot import New_taxtree_draw


class CorrelationFramework:
    def __init__(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        self.x = x.copy()
        self.y = y.copy()

        self.sc = SignificantCorrelation(self.x, self.y, **kwargs)
        self.correlation_tree = self.sc.get_real_correlations()
        self.plot = _CorrelationPlotter(self.sc, self.correlation_tree)


class _CorrelationPlotter:
    def __init__(self, significant_correlation, correlation_tree):
        self.significant_correlation = significant_correlation
        self.correlation_tree = correlation_tree

    def plot_positive_negative_bars(self, ax: Axes, percentile, last_taxonomic_levels_to_keep=2, **kwargs):
        significant_bacteria = self.significant_correlation.get_most_significant_coefficients(
            percentile=percentile)
        if last_taxonomic_levels_to_keep is not None:
            print(significant_bacteria.index)
            significant_bacteria.index = [delete_empty_taxonomic_levels(str(i))
                                          for i in significant_bacteria.index]
            significant_bacteria.index = [delete_suffix(str(i))
                                          for i in significant_bacteria.index]
            # significant_bacteria.index = [str([h[4:].strip("_").capitalize() for h in i.split(";")][-2:])
            #                                   .replace("[", "").replace("]", "").replace("\'", "") for i in
            #                               significant_bacteria.index]
            significant_bacteria.index = [
                str(','.join(str(i).split(';')[-last_taxonomic_levels_to_keep:])).replace(",", ", ")
                for i in significant_bacteria.index]
            significant_bacteria.index = [re.sub(r'\w_+', '', str(i))
                                          for i in significant_bacteria.index]

        return PP.plot_positive_negative_bars(ax, significant_bacteria, **kwargs)

    def clean_correlation_framework(self):
        lst = []
        for i in self.correlation_tree.index:
            if re.fullmatch(r'^(([^;]+);)+[^;]+$', str(i)):
                lst.append(i)
                print(i)
        self.correlation_tree = self.correlation_tree[lst]

    def plot_graph(self, ax: Axes, dict={}, folder=""):
        if not dict:
            dict = {"netural": "yellow", "positive": "green", "negative": "red", "treshold": 1.0}
        self.clean_correlation_framework()
        return draw_tree(ax, self.correlation_tree, dict=dict, folder=folder)

    def plot_real_and_shuffled_hist(self, ax: Axes, **kwargs):
        return PPR.plot_real_and_shuffled_hist(ax, self.significant_correlation.coeff_df['real'],
                                               self.significant_correlation.coeff_df.drop('real',
                                                                                          axis=1).values.flatten(),
                                               **kwargs)


def delete_empty_taxonomic_levels(i):
    splited = i.split(';')
    while re.search(r'^[a-z]_+\d*$', splited[-1]) is not None:
        splited = splited[:-1]
    i = ""
    for j in splited:
        i += j
        i += ';'
    i = i[:-1]
    return i

def delete_suffix(i):
    m = re.search(r'_+\d+$', i)
    if m is not None:
        i = i[:-(m.end()-m.start())]
    return i


# def use_corr_framwork(X: pd.DataFrame, y, title=None, folder="plots"):
#     cf = CorrelationFramework(X, y)
#
#     fig1 = plt.figure(figsize=(12, 6))
#     ax1 = plt.subplot(1, 2, 1)
#     ax2 = plt.subplot(1, 2, 2)
#
#     positive_dict = {'color': 'green', 'height': 0.8}
#     negative_dict = {'color': 'red', 'height': 0.8}
#     cf.plot.plot_positive_negative_baref use_corr_framwork(X: pd.DataFrame, y, title=None, folder="plots"):
#     cf = CorrelationFramework(X, y)
#
#     fig1 = plt.figure(figsize=(12, 6))
#     ax1 = plt.subplot(1, 2, 1)
#     ax2 = plt.subplot(1, 2, 2)
#
#     positive_dict = {'color': 'green', 'height': 0.8}
#     negative_dict = {'color': 'red', 'height': 0.8}
#     cf.plot.plot_positive_negative_bars(ax1, 1, positive_dict=positive_dict, negative_dict=negative_dict)
#     real_hist_dict = {'bins': 30, 'color': 'g', 'label': 'Real values', 'density': True, 'alpha': 0.5}
#     shuffled_hist_dict = {'bins': 30, 'color': 'b', 'label': 'Shuffled values', 'density': True, 'alpha': 0.5}
#     cf.plot.plot_real_and_shuffled_hist(ax2, real_hist_dict=real_hist_dict, shuffled_hist_dict=shuffled_hist_dict)
#
#     if title is not None:
#         fig1.suptitle(title.replace("_", " "), fontsize=16)
#
#     fig1.tight_layout()
#     fig1.savefig(f"{folder}/{title}.svg")

def use_corr_framwork(X: pd.DataFrame, y, dict={},title=None, folder=""):
    cf = CorrelationFramework(X, y)

    plt.close('all')
    fig1 = plt.figure(figsize=(24, 18))
    plt.subplots_adjust(top=0.94, bottom=0.07, left=0.1, right=0.97, hspace=0.3, wspace=0.3)
    grid = plt.GridSpec(4, 8)
    ax1 = plt.subplot(grid[0:2, 1:3])
    ax2 = plt.subplot(grid[2:5, 0:3])
    ax3 = plt.subplot(grid[:5, 3:10])


    positive_dict = {'height': 0.8}
    negative_dict = {'height': 0.8}
    positive_dict['color'] = dict['positive'] if 'positive' in dict else 'green'
    negative_dict['color'] = dict['negative'] if 'negative' in dict else 'red'
    treshold = dict['treshold'] if 'treshold' in dict else 1
    real_color = dict['real'] if 'real' in dict else 'g'
    random_Color = dict['random'] if 'random' in dict else 'b'
    print('ploting tree')
    cf.plot.plot_graph(ax3, folder=folder)
    cf.plot.plot_positive_negative_bars(ax1, 1, positive_dict=positive_dict, negative_dict=negative_dict)
    real_hist_dict = {'bins': 30, 'color': real_color, 'label': 'Real values', 'density': True, 'alpha': 0.5}
    shuffled_hist_dict = {'bins': 30, 'color': random_Color, 'label': 'Shuffled values', 'density': True, 'alpha': 0.5}
    cf.plot.plot_real_and_shuffled_hist(ax2, real_hist_dict=real_hist_dict, shuffled_hist_dict=shuffled_hist_dict)

    if title is not None:
        fig1.suptitle(title, fontsize=16)
    if folder != '' and title is not None:
        fig1.savefig(f"{folder}/{title}.svg")
    else:
        fig1.savefig("figure1.svg")

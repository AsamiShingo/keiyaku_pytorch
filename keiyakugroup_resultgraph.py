import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys
import os
import numpy as np
from pathlib import Path

def create_graph(title, datas, xlabel, ylabels1, ylabels2, savefile):
    xvalue = datas[xlabel].values
    yvalues1 = [ datas[label].values for label in ylabels1 ]
    yvalues2 = [ datas[label].values for label in ylabels2 ]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    for yvalue, ylabel in zip(yvalues1, ylabels1):
        ax1.plot(xvalue, yvalue, marker='*', label=ylabel)

    for yvalue, ylabel in zip(yvalues2, ylabels2):
        ax2.plot(xvalue, yvalue, marker='o', label=ylabel)

    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.01, step=0.1))
    handler1, label1 = ax1.get_legend_handles_labels()

    ax2.set_ylim(0, 1.5)
    ax2.set_yticks(np.arange(0, 1.51, step=0.1))
    handler2, label2 = ax2.get_legend_handles_labels()

    ax1.set_title(title)
    ax1.set_xticks(xvalue)
    ax1.legend(handler1 + handler2, label1 + label2, loc='upper left', borderaxespad=0.)
    
    plt.savefig(savefile)
    
args = sys.argv

savedir = ""
if len(args) >= 2:
    savedir = args[1]
else:
    dirpath = Path(os.path.join(os.path.dirname(__file__), "savedir"))
    dirs = list(dirpath.glob("??????????????_*"))
    target_dirs = { dir: os.stat(dir).st_mtime for dir in dirs if os.path.isdir(dir) == True }
    if len(target_dirs) > 0:
        savedir = max(target_dirs, key=target_dirs.get)

if os.path.isdir(savedir) != True:
    print("対象ディレクトリがありません(dir={})".format(savedir))
    sys.exit(9)

resultcsv = os.path.join(savedir, "result_data.csv")
if os.path.isfile(resultcsv) != True:
    print("読込対象ファイルがありません(dir={})".format(resultcsv))
    sys.exit(9)

df = pd.read_csv(resultcsv)

create_graph("文章グループ化", df, "epoch", ["val_output1_fvalue", "val_output1_precision", "val_output1_recall"], ["output1_loss", "val_output1_loss"], os.path.join(savedir, "result_graph1.png"))
create_graph("文章分類", df, "epoch", ["val_output2_fvalue", "val_output2_precision", "val_output2_recall"], ["output2_loss", "val_output2_loss"], os.path.join(savedir, "result_graph2.png"))
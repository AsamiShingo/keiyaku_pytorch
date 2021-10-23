from flask import Flask, url_for, redirect, send_file
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename
import os
import glob
import re
import threading
import json
import numpy as np
import datetime
from keiyakudata import KeiyakuData
from keiyakumodel import KeiyakuModel
from keiyakumodelfactory import KeiyakuModelFactory

DATA_DIR = os.path.join(os.path.dirname(__file__), r"data")
ANALYZE_DIR = os.path.join(os.path.dirname(__file__), r"analyze")
create_seqid_mutex = threading.Lock()
keiyaku_analyze_mutex = threading.Lock()

def keiyaku_analyze(seqid):
    keiyaku_analyze_mutex.acquire()

    targetdir = os.path.join(DATA_DIR, seqid)
    
    with open(os.path.join(targetdir, "param.json"), "r") as parafile:
        paradata = json.load(parafile)
        txtname = os.path.join(targetdir, paradata["txtname"])
        csvname = os.path.join(targetdir, paradata["csvname"])
    
    csvpath = os.path.join(targetdir, csvname)

    analyze_dir = os.path.join(ANALYZE_DIR, seqid)
    os.makedirs(analyze_dir, exist_ok=True)
    analyze_file = os.path.join(analyze_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".txt")

    keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel()

    keiyakudata = KeiyakuData(csvpath)
    datas = keiyakudata.get_datas()
    predict_datas = keiyakudata.get_group_datas(tokenizer, model.seq_len)

    np.set_printoptions(precision=2, floatmode='fixed')

    with open(analyze_file, "w") as f:
        for i in range(0, len(datas), 1000):
            targets = datas[i:i+1000]
            predict_targets = predict_datas[i:i+1000]
            scores = keiyakumodel.predict(predict_targets)

            for target, score1, score2 in zip(targets, scores[0], scores[1]):
                sentense = target[6]
                kind1 = score2.argmax()
                if score1 >= 0.5:
                    f.write("{}---------------------------------------------\n".format(score1))
                    
                f.write("{}-{:0.2f}:{}\n".format(kind1, score2[kind1], sentense))

    keiyaku_analyze_mutex.release()
    
    return analyze_file, txtname


def create_seqid():
    create_seqid_mutex.acquire()

    dirs = glob.glob(os.path.join(DATA_DIR, r"?????"))
    dirs = [os.path.basename(dir) for dir in dirs if os.path.isdir(dir)]
    seqs = [int(seq) for seq in dirs if re.match(r'[0-9]{5}', seq)]

    if len(seqs) != 0:
        seq = str(max(seqs) + 1)
    else:
        seq = "0"

    seqid = seq.zfill(5)
    os.mkdir(os.path.join(DATA_DIR, seqid))

    create_seqid_mutex.release()

    return seqid

def get_download_filepath(seqid):
    targetdir = os.path.join(DATA_DIR, seqid)
    
    with open(os.path.join(targetdir, "param.json"), "r") as parafile:
        paradata = json.load(parafile)
        filepath = os.path.join(targetdir, paradata["filename"])

    return filepath

def delete_seqid(seqid):
    dir = os.path.join(DATA_DIR, seqid)
    if os.path.isdir(dir) == False:
        raise FileNotFoundError("{}のディレクトリが存在しません".format(seqid))

    for delfile in os.listdir(dir):
        os.remove(os.path.join(dir, delfile))

    os.rmdir(dir)

def save_upload_file(seqid, request):
    f = request.files["file"]
    targetdir = os.path.join(DATA_DIR, seqid)

    uploadsavename = secure_filename(f.filename)
    txtname = "txt_{}.txt".format(os.path.basename(uploadsavename))
    csvname = "csv_{}.csv".format(os.path.basename(uploadsavename))
    uploadsavepath = os.path.join(targetdir, uploadsavename)
    txtpath = os.path.join(targetdir, txtname)
    csvpath = os.path.join(targetdir, csvname)
    f.save(uploadsavepath)
    KeiyakuData.create_keiyaku_data(uploadsavepath, txtpath, csvpath)

    with open(os.path.join(targetdir, "param.json"), "w") as parafile:
        data = {}
        data["filename"] = uploadsavename
        data["txtname"] = txtname
        data["csvname"] = csvname
        json.dump(data, parafile, ensure_ascii=False, indent=4)

def get_datas():
    datas = []
    dirs = glob.glob(os.path.join(DATA_DIR, r"?????"))
    for dir in dirs:
        filename = ""
        seqid = os.path.basename(dir)
        with open(os.path.join(dir, "param.json"), "r") as parafile:
            paradata = json.load(parafile)
            filename = paradata["filename"]

        if filename != "":
            datas.append((seqid, filename))

    return datas

app = Flask(__name__)

@app.route("/")
def index():
    datas = get_datas()
    return render_template("index.html", datas=datas)

@app.route("/upload", methods=["POST"])
def upload():
    seqid = create_seqid()
    save_upload_file(seqid, request)
    return redirect(url_for("index"))

@app.route("/download", methods=["POST"])
def download():
    seqid = request.form["seqid"]
    filepath = get_download_filepath(seqid)
    return send_file(filepath, as_attachment=True, attachment_filename=os.path.basename(filepath))

@app.route("/delete", methods=["POST"])
def delete():
    seqid = request.form["seqid"]
    delete_seqid(seqid)
    return redirect(url_for("index"))

@app.route("/analyze", methods=["POST"])
def analyze():
    seqid = request.form["seqid"]
    filepath, filename = keiyaku_analyze(seqid)
    return send_file(filepath, as_attachment=True, attachment_filename=filename)
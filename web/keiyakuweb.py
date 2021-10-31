from flask import Flask, url_for, redirect, send_file
from flask import request
from flask import jsonify
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

def keiyaku_analyze_txt(seqid):
    targetdir = os.path.join(DATA_DIR, seqid)    
    with open(os.path.join(targetdir, "param.json"), "r") as parafile:
        paradata = json.load(parafile)
        txtname = paradata["txtname"]
        csvname = paradata["csvname"]
    
    csvpath = os.path.join(targetdir, csvname)
    keiyakudata = KeiyakuData(csvpath)
    datas = keiyakudata.get_datas()
    scores1, scores2 = keiyaku_analyze(csvpath)

    analyze_dir = os.path.join(ANALYZE_DIR, seqid)
    os.makedirs(analyze_dir, exist_ok=True)
    analyze_file = os.path.join(analyze_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".txt")

    np.set_printoptions(precision=2, floatmode='fixed')

    with open(analyze_file, "w") as f:
        for data, score1, score2 in zip(datas, scores1, scores2):
            sentense = data[6]
            kind1 = score2.argmax()
            if score1 >= 0.5:
                f.write("{}---------------------------------------------\n".format(score1))
                
            f.write("{}-{:0.2f}:{}\n".format(kind1, score2[kind1], sentense))
    
    return analyze_file, txtname

def keiyaku_analyze_json(seqid):
    targetdir = os.path.join(DATA_DIR, seqid)    
    with open(os.path.join(targetdir, "param.json"), "r") as parafile:
        paradata = json.load(parafile)
        csvname = paradata["csvname"]
    
    csvpath = os.path.join(targetdir, csvname)
    scores1, scores2 = keiyaku_analyze(csvpath)
    
    jsondata = {}
    for col, score in enumerate(zip(scores1, scores2)):
        score1 = score[0]
        score2 = score[1]
        scoredata = {}
        scoredata[1] = round(float(score1[0]), 2)
        scoredata[2] = { i:round(float(score), 2) for i, score in enumerate(score2) }
        jsondata[col] = scoredata
        
    return jsondata

def keiyaku_analyze(csvpath):
    keiyaku_analyze_mutex.acquire()

    keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel()    
    keiyakudata = KeiyakuData(csvpath)
    predict_datas = keiyakudata.get_group_datas(tokenizer, model.seq_len)
    score1, score2 = keiyakumodel.predict(predict_datas)

    keiyaku_analyze_mutex.release()
    
    return score1, score2

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

def init_web(debugmode):
    if debugmode == False:
        KeiyakuModelFactory.get_keiyakumodel()
        
    app.run(debug=debugmode, host="0.0.0.0", port=80)
    
@app.route("/keiyaku_group/")
def index():
    datas = get_datas()
    return render_template("index.html", datas=datas)

@app.route("/keiyaku_group/upload", methods=["POST"])
def upload():
    seqid = create_seqid()
    save_upload_file(seqid, request)
    return redirect(url_for("index"))

@app.route("/keiyaku_group/download", methods=["POST"])
def download():
    seqid = request.form["seqid"]
    filepath = get_download_filepath(seqid)
    return send_file(filepath, as_attachment=True, attachment_filename=os.path.basename(filepath))

@app.route("/keiyaku_group/delete", methods=["POST"])
def delete():
    seqid = request.form["seqid"]
    delete_seqid(seqid)
    return redirect(url_for("index"))

@app.route("/keiyaku_group/analyze_txt", methods=["POST"])
def analyze_txt():
    seqid = request.form["seqid"]
    filepath, filename = keiyaku_analyze_txt(seqid)
    return send_file(filepath, as_attachment=True, attachment_filename=filename)

@app.route("/keiyaku_group/analyze_json", methods=["POST"])
def analyze_json():
    seqid = request.form["seqid"]
    jsondata = keiyaku_analyze_json(seqid)
    return jsonify(jsondata)
from flask import Flask, url_for, redirect, send_file
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename
import os
import glob
import re
import threading
import json
from ..keiyakudata import KeiyakuData

DATA_DIR = os.path.join(os.path.dirname(__file__), r"data")
create_seqid_mutex = threading.Lock()

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
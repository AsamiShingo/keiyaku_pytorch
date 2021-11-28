from flask import Flask, url_for, redirect, send_file
from flask import request
from flask import jsonify
from flask import render_template
from flask import flash
from flask import Blueprint
from werkzeug.utils import secure_filename
import werkzeug
import os
import glob
import re
import threading
import json
import numpy as np
import datetime
from keiyakudata import KeiyakuData
from keiyakumodelfactory import KeiyakuModelFactory

DATA_DIR = os.path.join(os.path.dirname(__file__), r"data")
ANALYZE_DIR = os.path.join(os.path.dirname(__file__), r"analyze")
UPLOAD_FILE_EXTENSION = [ ".pdf", ".doc", ".docx" ]
UPLOAD_FILE_MAX_SIZE_MB = 10

keiyaku_analyze_mutex = threading.Lock()

class KeiyakuWebData:
    
    PARA_FILE = "param.json"

    create_seqid_mutex = threading.Lock()        

    def __init__(self, seqid=None, orgfilename="", mimetype=""):
        if seqid != None:
            self.seqid = seqid
            self.paradata = {}

            with open(os.path.join(self.get_dirpath(), self.PARA_FILE), "r") as parafile:
                para = json.load(parafile)
                self.paradata["orgfilename"] = para["orgfilename"]
                self.paradata["filename"] = para["filename"]
                self.paradata["txtname"] = para["txtname"]
                self.paradata["csvname"] = para["csvname"]
                self.paradata["mimetype"] = para["mimetype"]
        else:
            if orgfilename == "" or mimetype == "":
                raise AttributeError()

            self.seqid = self._create_seqid()
            self.paradata = {}
            
            basefilename = os.path.basename(secure_filename(orgfilename))
            extension = os.path.splitext(orgfilename)[1]
            filename = "{}_{}.{}".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), basefilename, extension)
            self.paradata["orgfilename"] = orgfilename
            self.paradata["filename"] = filename
            self.paradata["txtname"] = "txt_{}.txt".format(os.path.basename(filename))
            self.paradata["csvname"] = "csv_{}.csv".format(os.path.basename(filename))
            self.paradata["mimetype"] = mimetype
            with open(os.path.join(self.get_dirpath(), self.PARA_FILE), "w") as parafile:
                json.dump(self.paradata, parafile, ensure_ascii=False, indent=4)

    def get_dirpath(self):
        return os.path.join(DATA_DIR, self.seqid)

    def get_orgfilename(self):
        return self.paradata["orgfilename"]

    def get_orgtxtname(self):
        return os.path.basename(self.get_orgfilename()) + ".txt"

    def get_filename(self):
        return self.paradata["filename"]

    def get_filepath(self):
        return os.path.join(self.get_dirpath(), self.get_filename())

    def get_txtname(self):
        return self.paradata["txtname"]

    def get_txtpath(self):
        return os.path.join(self.get_dirpath(), self.get_txtname())

    def get_csvname(self):
        return self.paradata["csvname"]

    def get_csvpath(self):
        return os.path.join(self.get_dirpath(), self.get_csvname())

    def get_mimetype(self):
        return self.paradata["mimetype"]

    def create_analyzepath(self):
        analyze_dir = os.path.join(ANALYZE_DIR, self.seqid)
        os.makedirs(analyze_dir, exist_ok=True)
        return os.path.join(analyze_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ".txt")

    def _create_seqid(self):
        self.create_seqid_mutex.acquire()

        dirs = glob.glob(os.path.join(DATA_DIR, r"?????"))
        dirs = [os.path.basename(dir) for dir in dirs if os.path.isdir(dir)]
        seqs = [int(seq) for seq in dirs if re.match(r'[0-9]{5}', seq)]
        seq = str(max(seqs) + 1) if len(seqs) != 0 else "0"

        seqid = seq.zfill(5)
        os.mkdir(os.path.join(DATA_DIR, seqid))

        self.create_seqid_mutex.release()

        return seqid

def keiyaku_analyze(csvpath):
    keiyaku_analyze_mutex.acquire()

    keiyakumodel, model, tokenizer = KeiyakuModelFactory.get_keiyakumodel()    
    keiyakudata = KeiyakuData(csvpath)
    predict_datas = keiyakudata.get_group_datas(tokenizer, model.seq_len)
    score1, score2 = keiyakumodel.predict(predict_datas)

    keiyaku_analyze_mutex.release()
    
    return score1, score2

view_app = Blueprint("view", __name__, static_url_path='/keiyaku_group/view', static_folder='./view/build')
app = Flask(__name__)
app.register_blueprint(view_app)

app.config["SECRET_KEY"] = "keiyaku_group_cosmo"
app.config['MAX_CONTENT_LENGTH'] = UPLOAD_FILE_MAX_SIZE_MB * 1024 * 1024

def init_web(debugmode):
    if debugmode == False:
        KeiyakuModelFactory.get_keiyakumodel()
        
    app.run(debug=debugmode, host="0.0.0.0", port=80)
    
@app.route("/keiyaku_group/", methods=["GET"])
def index():    
    datas = []
    for dir in glob.glob(os.path.join(DATA_DIR, r"?????")):
        seqid = os.path.basename(dir)
        data = KeiyakuWebData(seqid)

        if os.path.isfile(data.get_filepath()):
            datas.append((seqid, data.get_orgfilename()))

    return render_template("index.html", datas=datas)

@app.route("/keiyaku_group/api", methods=["GET"])
def api():
    return view_app.send_static_file("index.html")

@app.route("/keiyaku_group/api/list", methods=["GET"])
def api_list():
    result={"data": [], "code": 0}
    for dir in glob.glob(os.path.join(DATA_DIR, r"?????")):
        seqid = os.path.basename(dir)
        data = KeiyakuWebData(seqid)

        if os.path.isfile(data.get_filepath()):
            result["data"].append({"seqid": seqid, "filename": data.get_orgfilename()})

    return jsonify(result)

@app.route("/keiyaku_group/upload", methods=["POST"])
def upload():
    try:
        f = request.files["file"]
    except werkzeug.exceptions.RequestEntityTooLarge:
        flash("ファイルサイズが{}MBを超えるためアップロードできません".format(UPLOAD_FILE_MAX_SIZE_MB), category="flash_error")
        return redirect(url_for("index"))

    data = KeiyakuWebData(orgfilename=f.filename, mimetype=request.mimetype)
    
    extension = os.path.splitext(f.filename)[1].lower()
    if extension not in UPLOAD_FILE_EXTENSION:
        flash("拡張子{}はアップロードできません".format(extension if extension != "" else "無し"), category="flash_error")
    else:
        f.save(data.get_filepath())
        KeiyakuData.create_keiyaku_data(data.get_filepath(), data.get_txtpath(), data.get_csvpath())

    return redirect(url_for("index"))

@app.route("/keiyaku_group/api/upload", methods=["POST"])
def api_upload():
    result={"data" : { "seqid":-1, "filename": ""}, "code": 0, "message": [] }

    try:
        f = request.files["file"]
    except werkzeug.exceptions.RequestEntityTooLarge:
        result["message"].append({"category": "error", "message": "ファイルサイズが{}MBを超えるためアップロードできません".format(UPLOAD_FILE_MAX_SIZE_MB)})
        result["code"] = 9
        return jsonify(result)

    data = KeiyakuWebData(orgfilename=f.filename, mimetype=request.mimetype)
    
    extension = os.path.splitext(f.filename)[1].lower()
    if extension not in UPLOAD_FILE_EXTENSION:
        result["message"].append({"category": "error", "message": "拡張子{}はアップロードできません".format(extension if extension != "" else "無し")})
        result["code"] = 9
    else:
        f.save(data.get_filepath())
        KeiyakuData.create_keiyaku_data(data.get_filepath(), data.get_txtpath(), data.get_csvpath())
        result["data"]["seqid"] = data.seqid
        result["data"]["filename"] = data.get_orgfilename()
        result["message"].append({"category": "info", "message": "{}をアップロードしました".format(data.get_orgfilename())})

    return jsonify(result)

@app.route("/keiyaku_group/download", methods=["POST"])
def download():
    seqid = request.form["seqid"]
    data = KeiyakuWebData(seqid)
    return send_file(data.get_filepath(), as_attachment=True, attachment_filename=data.get_orgfilename())

@app.route("/keiyaku_group/api/download", methods=["POST"])
def api_download():
    return download()

@app.route("/keiyaku_group/download_txt", methods=["POST"])
def download_txt():
    seqid = request.form["seqid"]
    data = KeiyakuWebData(seqid)
    return send_file(data.get_txtpath(), as_attachment=True, attachment_filename=data.get_orgtxtname())

@app.route("/keiyaku_group/api/download_txt", methods=["POST"])
def api_download_txt():
    return download_txt();

@app.route("/keiyaku_group/delete", methods=["POST"])
def delete():
    seqid = request.form["seqid"]
    data = KeiyakuWebData(seqid)

    dirpath = data.get_dirpath()
    if os.path.isdir(dirpath) == True:
        for delfile in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, delfile))

        os.rmdir(dirpath)

    return redirect(url_for("index"))

@app.route("/keiyaku_group/api/delete", methods=["POST"])
def api_delete():
    seqid = request.form["seqid"]
    result={"data" : { "seqid": seqid }, "code": 0, "message": [] }
    
    data = KeiyakuWebData(seqid)

    dirpath = data.get_dirpath()
    if os.path.isdir(dirpath) == True:
        for delfile in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, delfile))

        os.rmdir(dirpath)

    return jsonify({})

@app.route("/keiyaku_group/analyze", methods=["POST"])
def analyze():
    seqid = request.form["seqid"]
    data = KeiyakuWebData(seqid)
    
    scores1, scores2 = keiyaku_analyze(data.get_csvpath())
    keiyakudata = KeiyakuData(data.get_csvpath())
    sentensedatas = keiyakudata.get_datas()
    analyze_path = data.create_analyzepath()

    np.set_printoptions(precision=2, floatmode='fixed')
    with open(analyze_path, "w") as f:
        for sentensedata, score1, score2 in zip(sentensedatas, scores1, scores2):
            sentense = sentensedata[6]
            kind1 = score2.argmax()
            if score1 >= 0.5:
                f.write("{}---------------------------------------------\n".format(score1))
                
            f.write("{}-{:0.2f}:{}\n".format(kind1, score2[kind1], sentense))

    return send_file(analyze_path, as_attachment=True, attachment_filename=data.get_orgtxtname())

@app.route("/keiyaku_group/api/analyze", methods=["POST"])
def api_analyze():
    seqid = request.form["seqid"]
    result={"data" : {}, "code": 0, "message": [] }
    
    data = KeiyakuWebData(seqid)

    scores1, scores2 = keiyaku_analyze(data.get_csvpath())
    
    jsondata = {}
    for col, score in enumerate(zip(scores1, scores2)):
        score1 = score[0]
        score2 = score[1]
        scoredata = {}
        scoredata[1] = round(float(score1[0]), 2)
        scoredata[2] = { i:round(float(score), 2) for i, score in enumerate(score2) }
        jsondata[col] = scoredata

    result["data"] = jsondata
    return jsonify(result)
    
@app.after_request
def after_request(response):
    if app.debug:
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
    return response
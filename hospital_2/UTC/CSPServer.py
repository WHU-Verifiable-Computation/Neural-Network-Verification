import json
import os
import time
import zipfile
from oprtDB import oprtDB
import flask
import pandas as pd
import requests
from flask import Flask, request, render_template

from CSP import CSP
from TPAServer import tpa

app = Flask(__name__)


csp = CSP()


@app.route('/CSP/init', methods=['POST'])
def init():
    print(csp.filesList)
    return 'Start Running'


@app.route('/CSP/upload', methods=['GET', 'POST'])
def upload_file():
    # user input fid&uid in order to build the dynamic path
    uid = flask.request.form.get("user_id")
    fid = flask.request.form.get("file_id")
    optType = flask.request.form.get("type")
    optPos = flask.request.form.get("number")


    # store all the files that is post
    if request.method == 'POST':
        fSig = request.files.getlist("sig")
        fFile = request.files.getlist("file")

        csp_path = r'/home/andy/UTCfiles/CSP/'
        path = csp_path + fid + uid + ".csv"
        o = oprtDB()

        if len(fSig) != len(fFile):
            return 'fileNum != sigNum ERROR'
        lenth = len(fSig)
        dir = uid + fid
        sigStorePath = dir + '/sigStore'
        fileStorePath = dir + '/fileStore'
        isExists = os.path.exists(csp_path+dir)
        basedir = os.path.dirname(os.path.dirname(__file__))
        if isExists and (optType == 'upload'):
            # store_files(basedir, fSig, fFile, sigStorePath, fileStorePath)
            #
            # filesList = csp.extragen_fileList(dir, lenth, fSig, fFile, basedir, sigStorePath, fileStorePath)
            # csp.filesDict[dir] = filesList
            # print(filesList)
            # print(csp.filesDict)
            return 'xu shang le'
        elif isExists and (optType == 'insert'):
            #####store_files(basedir, fSig, fFile, sigStorePath, fileStorePath)
            ##### filesDict = csp.insert_blockList(optPos, lenth, fSig, fFile, basedir, sigStorePath, fileStorePath, dir)
            VSig = ""
            TFile = ""
            for file_name in fSig:
                filename = file_name.filename.split('/')
                temp_path = os.path.join(csp_path, sigStorePath, filename[0])
                file_name.save(temp_path)
                VSig = temp_path
            for file_name in fFile:
                filename = file_name.filename.split('/')
                temp_path = os.path.join(csp_path, fileStorePath, filename[0])
                file_name.save(temp_path)
                TFile = temp_path
            o.insertCSV(optPos,path,VSig,TFile)
            return 'block inserted successfully'
        elif isExists and (optType == 'delete'):
            #####filesDict = csp.delete_blockList(optPos, dir)
            data = pd.read_csv(path).loc[int(optPos)]
            os.remove(data[0])
            os.remove(data[1])
            o.deleteCSV(optPos,path)
            return 'block deleted successfully'
        elif isExists and (optType == 'update'):
            # store_files(basedir, fSig, fFile, sigStorePath, fileStorePath)
            # filesDict = csp.update_blockList(optPos, lenth, fSig, fFile, basedir, sigStorePath, fileStorePath, dir)
            VSig = ""
            TFile = ""
            for file_name in fSig:
                filename = file_name.filename.split('/')
                temp_path = os.path.join(csp_path, sigStorePath, filename[0])
                file_name.save(temp_path)
                VSig = temp_path
            for file_name in fFile:
                filename = file_name.filename.split('/')
                temp_path = os.path.join(csp_path, fileStorePath, filename[0])
                file_name.save(temp_path)
                TFile = temp_path
            o.changeCSV(optPos,path,VSig,TFile)
            return 'block updated successfully'
        elif isExists and (optType == 'fileDelete'):
            filesDict = csp.delete_flieList(basedir, dir)
            print(filesDict)
            return 'file is deleted successfully'
        elif isExists and (optType == 'fileUpdate'):
            filesDict = csp.update_flieList(lenth, fSig, fFile, basedir, sigStorePath, fileStorePath, dir)
            os.makedirs(sigStorePath)
            os.makedirs(fileStorePath)
            store_files(basedir, fSig, fFile, sigStorePath, fileStorePath)
            print(filesDict)
            return 'file is updated successfully'
        elif not isExists:
            os.makedirs(csp_path+sigStorePath)
            os.makedirs(csp_path+fileStorePath)
            store_files(csp_path, fSig, fFile, sigStorePath, fileStorePath)

            # filesList =
            csp.gen_fileList(csp_path,dir,basedir, sigStorePath, fileStorePath)
            # csp.filesDict[dir] = filesList
            # print(filesList)
            # print(csp.filesDict)
            return 'block uploaded successfully'
    else:
        return render_template('upload.html')


def find_signature(chal, dir):
    assert isinstance(chal, dict)
    csp_path = r'/home/andy/UTCfiles/CSP/'
    path = dir + ".csv"
    csp_path = csp_path + path
    # construct sig based on chal & sig files
    sig = list()
    for i in chal.keys():
        data = pd.read_csv(csp_path).loc[int(i)]
        sigDir = data[0]
        #print(sigDir)
        sigDir = sigDir.split('/')[-1]
        sigDir = '/home/andy/UTCfiles/CSP/'+dir+'/sigStore/'+sigDir

        f = open(sigDir, "a+")
        f.seek(0)
        signature = f.readline()
        sig.append(tpa.group.deserialize(signature.encode()))
    f.close()
    #print(sig)

    return sig


def store_files(csp_path, fSig, fFile, sigStorePath, fileStorePath):
    for zipfile in fSig:
        filename = zipfile.filename.split('/')
        path = os.path.join(csp_path, sigStorePath, filename[0])
        zipfile.save(path)

    sigOutputDir = csp_path + sigStorePath + '/'
    unzipFile(path, sigOutputDir)
    os.remove(path)

    for zipfile in fFile:
        filename = zipfile.filename.split('/')
        path = os.path.join(csp_path, fileStorePath, filename[0])
        zipfile.save(path)

    fileOutPutDir = csp_path + fileStorePath + '/'
    unzipFile(path, fileOutPutDir)
    os.remove(path)


def unzipFile(path, outPutDir):
    f = zipfile.ZipFile(path, 'r')  # 压缩文件位置
    for file in f.namelist():
        f.extract(file, outPutDir)  # 解压位置
    f.close()


@app.route('/CSP/genproof', methods=['POST'])
def gen_proof():
    # recv message: get the chal & sig
    chalMessage = request.get_data()
    c = json.loads(chalMessage)
    uidfid = c['id']
    challenge = c['chalInfo']

    # recv message: get the challenge from json
    chal = dict()
    for key in challenge.keys():
        chal[key] = tpa.group.deserialize(challenge[key].encode())

    # # recv message: get the signature from json
    sig = find_signature(chal, uidfid)

    # gen proof
    t0 = time.perf_counter()
    #print(chal)
    proof = csp.Proof(chal, sig, uidfid)
    t1 = time.perf_counter()
    print("spend time is:", (t1 - t0))

    print(proof)


    # send message: construct message
    proofMessage = dict()
    p = dict()
    for key in proof.keys():
        p[key] = tpa.group.serialize(proof[key]).decode()
    proofMessage['pInfo'] = p
    proofMessage['chalInfo'] = challenge
    proofMessage['uidfid'] = uidfid

    # send message : to TPA
    tpaUrl = 'http://127.0.0.1:6000/TPA/verify'
    p = requests.post(tpaUrl, data=json.dumps(proofMessage))
    return "proof ve sent"





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000)
    #csp_path = r'/home/andy/UTCfiles/CSP/22.csv'
    #filename = csp_path.split('/')[-1]
    #print(filename)



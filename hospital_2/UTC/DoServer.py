import requests
from flask import Flask, request, render_template

import dvd_blk
import head
import toolsForSystem
from Do import Do


app = Flask(__name__)
do = Do(0, head.userParam)

#http://127.0.0.1:8000/TPA/challenge
@app.route('/DO/taggen', methods=['POST'])
def TagGen():
    do.fid = input("input fid: ")
    do.uid = input("input uid: ")
    do._blockSize, do._blockNum, do._size, do.dlit, do.filePath = dvd_blk.split_By_size(do.fid, do.uid)
    #do._blockSize, do._blockNum, do._size, do.dlit, do.filePath = toolsForSystem.split_By_row(do.fid, do.uid)
    do.TagGen()
    return 'Start Running'


@app.route('/DO/2tpa', methods=['POST'])
def do2tpa():
    # get the verInfos
    do.verInfos = [do.dlit.getVerifyInfo(do.fid, do.uid, j) for j in range(do._blockNum)]#8
    VT = do.verInfos


    # 给TPA服务器传请求
    do.url = 'http://0.0.0.0:8000/TPA/init'
    do.header = {'Content-Type': 'application/json'}
    # 这句非常重要，有这句代码才能表示传参是json格式

    do.data = {
        "fid": do.fid,#9
        "uid": do.uid,#10
        "VT": VT,
    }
    Do.res = requests.post(do.url, headers=do.header, json=do.data)
    return 'information sent'


@app.route('/DO/changedlit', methods=['POST'])
def changeDlit():
    do.fid = input("input fid: ")
    do.uid = input("input uid: ")
    type = input("input the operation(number): ")
    type = int(type)

    url = 'http://0.0.0.0:8000/TPA/dlitHandler'
    header = {'Content-Type': 'application/json'}
    # 这句非常重要，有这句代码才能表示传参是json格式
    i = input("input the block number that you want to change: ")
    tpaMessage = dict()

    if type == 1:
        #insert block
        VT = (1, 1665149757.7191594)
        tpaMessage = {
            "type": type,
            "fid": do.fid,#11
            "uid": do.uid,#12
            "i": i,
            "VT": VT,
        }
    elif type == 2:
        # delete block
        tpaMessage = {
            "type": 2,
            "fid": do.fid,#13
            "uid": do.uid,#14
            "i": i,
        }

    elif type == 3:
        #recover block
        VT = (1, 1665149757.7191594)
        tpaMessage = {
            "type": type,
            "fid": do.fid,#15
            "uid": do.uid,#16
            "i": i,
            "VT": VT,
        }

    elif type == 4:
        #deleteFile
        tpaMessage = {
            "type": type,
            "fid": do.fid,#15
            "uid": do.uid,#16
        }
        #############################
    elif type == 5:
        #modifyFile
        tpaMessage = {
            "type": type,
            "fid": do.fid,#15
            "uid": do.uid,#16
            "VT": do.verInfos,
        }

    Do.res = requests.post(url, headers=header, json=tpaMessage)
    return 'operation succeed'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000)
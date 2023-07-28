import json
import time


from UTC.TPA import TPA
from flask import Flask, request
import requests
from UTC.oprtDB import oprtDB
from UTC.head import User2TPAMessageType
from UTC.dlit import DLIT


app = Flask(__name__)

tpa = TPA()
global fid, uid


@app.route('/TPA/test', methods=['POST','get'])
def test():
    print(bool)
    return "init success"


@app.route('/TPA/init', methods=['POST'])
def init():
    data = request.get_json()
    fid = data['fid']
    uid = data['uid']
    isExist = tpa.dlit
    if isExist is None:
        tpa.dlit = DLIT(data['fid'], data['uid'], data['VT'])
    else:
        if tpa.dlit.findFileNode(fid, uid)[1] is None:
            tpa.dlit.insertFile(data['fid'], data['uid'], data['VT'])
            print("new file stored")
        else:
            tpa.dlit.modifyFile(data['fid'], data['uid'], data['VT'])
            print("Files have been already stored, change blocks instead of files plz")



    tpa.verInfos = data['VT']
    print(tpa.verInfos)
    print(tpa.dlit.printDLIT())
    print(data['VT'])
    return "init success"


#message
#type
#fid
#uid
#i
#VT
@app.route('/TPA/dlitHandler', methods=['POST'])
def dlitHandler():
    data = request.get_json()
    path = "/home/andy/UTCfiles/TPA/" + data['fid'] + data['uid'] + ".csv"
    o = oprtDB()
    # print(path)
    if data['type'] == User2TPAMessageType.INSERTBLOCK.value:
        V = data['VT'][0]
        T = data['VT'][1]
        ###############################################################
        #tpa.dlit.insertBlock(data['fid'], data['uid'], data['i'], V, T)
        o.insertCSV(data['i'],path,V,T)
        ###############################################################
        # tpa.verInfos.insert(int(data['i']),[V, T])
        #print(tpa.dlit.printDLIT())
        print("insertblock success")
        return "insertblock success"
    elif data['type'] == User2TPAMessageType.DELETEBLOCK.value:
        #tpa.dlit.deleteBlock(data['fid'], data['uid'], data['i'])
        o.deleteCSV(data['i'],path)
        # tpa.verInfos.pop(int(data['i']))
        #print(tpa.dlit.printDLIT())
        print("deleteblock success")
        return "deleteblock success"
    elif data['type'] == User2TPAMessageType.MODIFYBLOCK.value:
        V = data['VT'][0]
        T = data['VT'][1]
        #tpa.dlit.modifyBlock(data['fid'], data['uid'], data['i'], V, T)
        o.changeCSV(data['i'],path,V,T)
        # tpa.verInfos[int(data['i'])][0] = V
        # tpa.verInfos[int(data['i'])][1] = T
        #print(tpa.dlit.printDLIT())
        print("modifyblock success")
        return "modifyblock success"
    elif data['type'] == User2TPAMessageType.DELETEFILE.value:
        tpa.dlit.deleteFile(data['fid'], data['uid'])
        print(tpa.dlit.printDLIT())
        print('deletefile success')
        return "deletefile success"
    elif data['type'] == User2TPAMessageType.MODIFYFILE.value:
        tpa.dlit.modifyFile(data['fid'], data['uid'], data['VT'])
        print(tpa.dlit.printDLIT())
        print('modifyfile success')
        return "modifyfile success"
    # elif data['type'] == User2TPAMessageType.INSERTFILE.value:
    #     tpa.dlit.insertFile(data['fid'], data['uid'], data['VT'])
    #     return "inserfile success"
    return "nothing happened"



@app.route('/TPA/getdlitinfo', methods=['GET'])
def getdlitinfo():
    ans = ""
    cur = tpa.dlit.head
    while cur:
        blocks = cur.blocks
        while blocks:
            ans += blocks.V
            ans += blocks.T
            ans += '\n'
            blocks = blocks.next
        cur = cur.next
    return ans


@app.route('/TPA/challenge', methods=['POST', 'GET'])
def challenge(fid,uid,metdNum,CNum,bool=False, num=-1):
    uidfid = str(fid) + str(uid)

    #construct message
    challMessage = dict()
    #blockNum = tpa.dlit.getBlockCount(1,1)
    #metdNum = input("input the methodNum of challenge(0,1,2):")

    challenge = tpa.challenge(fid, uid, metdNum, CNum, bool, num)#17
    # print('--------------list------------')
    # print(len(challenge))
    challMessage['id'] = uidfid
    if int(metdNum) == 2:
        tempList = challenge
        for i in range(len(tempList)):
            challenge = tempList[i]

            c = {}
            for key in challenge.keys():
                c[key] = tpa.group.serialize(challenge[key]).decode()
            challMessage['chalInfo'] = c
            print(c)

            # send to css
            cssUrl = 'http://127.0.0.1:9000/CSP/genproof'
            r = requests.post(cssUrl, data=json.dumps(challMessage))
            print( str(i+1) + " times verified")
    else:
        c = {}
        for key in challenge.keys():
            c[key] = tpa.group.serialize(challenge[key]).decode()
        challMessage['chalInfo'] = c
        # print(c)

        # send to css
        cssUrl = 'http://127.0.0.1:9000/CSP/genproof'
        r = requests.post(cssUrl, data=json.dumps(challMessage))

    return 'chal ve sent'


@app.route('/TPA/verify', methods=['POST', 'GET'])
def verify():
    #t0 = time.perf_counter()
    #_________________________________________send proof
    c2 = request.get_data()
    c2 = json.loads(c2)
    proof = c2['pInfo']
    for key in proof.keys():
        proof[key] = tpa.group.deserialize(proof[key].encode())
    print(proof)
    #_________________________________________send proof

    challenge = c2['chalInfo']
    for key in challenge.keys():
        challenge[key] = tpa.group.deserialize(challenge[key].encode())

    uidfid = c2['uidfid']
    if tpa.verify(challenge, proof, uidfid):
        print("verified!")
        bool = True


    else:
        print("failed")
        bool = False

    with open(r"result.txt", "w", encoding='utf-8') as f:
        f.write(str(bool))

    return 'done'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)


import math, time, binascii
import requests
import pickle
import head
from TPA import TPA
from charm.toolbox.pairinggroup import ZR, G1
from flask import Flask, request
import socket, os

from myToolBox import zipFile

app = Flask(__name__)
tpa=TPA()
class Do:
    def __init__(self, taskNum, userParam):
        # fid = input("input fid: ")#4
        # uid = input("input uid: ")#5
        # self.fid = fid
        # self.uid = uid
        # self._taskNum = taskNum
        # self._blockSize,self._blockNum,self._size,self.dlit = dvd_blk.split_By_size(fid, uid)#6

        self.fid = int
        self.uid = int
        self._taskNum = taskNum

        self._blockSize = None
        self._blockNum = None
        self._size = None
        self.dlit = None
        self.filePath = None

        self._tags = []
        self.verInfos = []
        self.group = head.group
        self.g = head.g
        self._x = userParam['x']
        self.v = self.g ** self._x
        self.u = userParam['u']

        self.fileName = 'test0'


    def TagGen(self):

        #store and upload to csp
        self.fnameSig = self.group.hash(self.fileName, G1) ** self._x
        print("blocknum is",self._blockNum)
        self.verInfos = [self.dlit.getVerifyInfo(self.fid,self.uid,i) for i in range(self._blockNum)]#7
        # print("verinfo:",self.verInfos)
        signatures = list()
        sub = 0
        # new = os.getcwd() + '/test/sigStore/'
        new = '/home/andy/UTCfiles/test/sigStore/'

        if not os.path.exists(new):  # 判断当前路径是否存在，没有则创建new文件夹
            os.makedirs(new)

        for i in range(self._blockNum):
            f = open(self.filePath[i], 'rb')
            # f.seek(self._taskNum + i * self._blockSize, 0)
            f.seek(0)
            blockBytes = f.read()
            f.close()

            mi = blockBytes
            mi = int(binascii.hexlify(mi), 16)
            mi = self.group.init(ZR, mi)
            etr = self.dlit.getVerifyInfo(self.fid,self.uid,i)#16
            a1 = str(etr[0])+str(etr[1])
            sig = self.group.hash(a1, G1) * self.u ** mi
            sig = sig ** self._x
            print(str(i) + "sig is ",sig)
            signame = new + 'sig' + '_' + str(i)

            nw = new + 'FAQ/'

            # if not os.path.exists(nw):  # 判断当前路径是否存在，没有则创建new文件夹
            #     os.makedirs(nw)
                
            sgname = nw +'sig'+'_'+str(i)

            sg={}
            sg[i] = tpa.group.serialize(sig)
            with open(signame, 'wb') as f:
                f.write(sg[i])


            sgn={}
            sgn[i] = tpa.group.deserialize(sg[i])
            # with open(sgname, 'w') as f:
            #     f.write(str(sgn[i]))
            
            signatures.append(sig)

        self._tags = signatures
        print("sg is ",sg)
        print("sgn is ",sgn)

        zipFile('/home/andy/UTCfiles/test/fileStore', '/home/andy/UTCfiles/test/fileStore.zip')
        zipFile('/home/andy/UTCfiles/test/sigStore', '/home/andy/UTCfiles/test/sigStore.zip')

        return self._blockSize, self._blockNum, self._size

    def TagGen4Update(self):
        # store and upload to csp
        self.fnameSig = self.group.hash(self.fileName, G1) ** self._x
        print("blocknum is", self._blockNum)
        self.verInfos = [self.dlit.getVerifyInfo(1, 1, i) for i in range(self._blockNum)]
        # print("verinfo:", self.verInfos)
        signatures = list()
        sub = 0
        new = os.getcwd() + '/test/'

        if not os.path.exists(new):  # 判断当前路径是否存在，没有则创建new文件夹
            os.makedirs(new)
        for i in range(self._blockNum):
            f = open(head.testDatafilePathDO, 'rb')
            f.seek(self._taskNum + i * self._blockSize, 0)
            blockBytes = f.read(self._blockSize)
            f.close()

            mi = blockBytes
            mi = int(binascii.hexlify(mi), 16)
            mi = self.group.init(ZR, mi)

            etr = self.dlit.getVerifyInfo(1, 1, i)
            a1 = str(etr[0]) + str(etr[1])
            sig = self.group.hash(a1, G1) * self.u ** mi
            sig = sig ** self._x
            print(str(i) + "sig is ", sig)
            signame = new + 'sig' + '_' + str(i)

            nw = new + 'FAQ/'

            if not os.path.exists(nw):  # 判断当前路径是否存在，没有则创建new文件夹
                os.makedirs(nw)

            sgname = nw + 'sig' + '_' + str(i)

            sg = {}
            sg[i] = tpa.group.serialize(sig)
            with open(signame, 'wb') as f:
                f.write(sg[i])

            sgn = {}
            sgn[i] = tpa.group.deserialize(sg[i])
            with open(sgname, 'w') as f:
                f.write(str(sgn[i]))

            signatures.append(sig)

        self._tags = signatures
        print("sg is ", sg)
        print("sgn is ", sgn)
        return self._blockSize, self._blockNum, self._size



    def fileRead(read_path):
        # 打开文件
        result_list = []
        print(read_path)
        file_data = open(read_path, encoding='utf8')
        file_read_data = file_data.read()
        # 获得列表并去重
        file_data_list = list(set(reg_name.findall(file_read_data)))
        # 数据正则数据处理
        for result in file_data_list:
            result_list.append(result.replace("'", ''))
               # 关闭文件
        file_data.close()
        return result_list





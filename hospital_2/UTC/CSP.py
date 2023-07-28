import binascii
import csv
import json
import linecache
import os
import shutil
import time

import pandas as pd
import requests

import head
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT, pair


class CSP:
    def __init__(self):
        self.group = head.group
        self.csp_path = r'/home/andy/UTCfiles/CSP/'
        # store the filesInfo into list to finally store in Dict
        filesDict = dict()
        self.filesDict = filesDict
        # filesList = list()
        # self.filesList = filesList


    def Proof(self, chal, sig, dir):
        sigCount = 0
        assert isinstance(chal, dict)
        proof = dict()

        M = self.group.init(ZR, 0)
        T = self.group.init(G1, 1)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #filesList = self.filesDict[dir]
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #assert isinstance(filesList, list)
        #genju dir huode csv wenjian
        csv_name = dir + ".csv"
        #csv_name = "large11.parquet"
        t0 = time.perf_counter()
        for i in chal.keys():
            data = pd.read_csv(self.csp_path+csv_name).loc[int(i)]
            #data = pd.read_parquet(self.csp_path+csv_name, engine="fastparquet").loc[int(i)]
            fileDir = data[1]
            #fileDir = filesList[int(i)][1]
            fileDir = fileDir.split('/')[-1]
            fileDir = '/home/andy/UTCfiles/CSP/'+dir+'/fileStore/'+fileDir
            # print(fileDir)
            f = open(fileDir, 'rb')
            f.seek(0)
            blockBytes = f.read()
            f.close()

            mi = blockBytes
            tempStore = binascii.hexlify(mi) #gai 1  //zhuanhuanweierjinzhi
            mi = int(tempStore, 16) #gai 2          //zhuanhuanweijinzhi
            mi = self.group.init(ZR, mi)

            M += mi * chal[i]
            T *= sig[sigCount] ** chal[i]
            sigCount += 1

        proof['T'] = T
        proof['M'] = M
        t1 = time.perf_counter()
        print("spend time is:", (t1 - t0))

        return proof

    # former method
    # def gen_fileList(self, lenth, fSig, fFile, basedir, sigStore, fileStore):
    #     filesList = list()
    #     for i in range(lenth):
    #         tempList = list()
    #         sigName = fSig[i].filename.split('/')
    #         fileName = fFile[i].filename.split('/')
    #         pathSig = os.path.join(basedir, 'UTC(demo2)', sigStore, sigName[0])
    #         pathFile = os.path.join(basedir, 'UTC(demo2)', fileStore, fileName[0])
    #         tempList.append(pathSig)
    #         tempList.append(pathFile)
    #         filesList.append(tempList)
    #
    #     return filesList

    def gen_fileList(self, csp_path, dir,basedir, sigStore, fileStore):
        filesList = list()
        path_sig = os.path.join(csp_path, sigStore)
        path_file = os.path.join(csp_path, fileStore)

        siglist = self.getFileList(path_sig)
        filelist = self.getFileList(path_file)

        lenth = len(self.getFileList(path_sig))
        for i in range(lenth):
            tempList = list()
            pathSig = siglist[i]
            pathFile = filelist[i]

            tempList.append(pathSig)
            tempList.append(pathFile)
            filesList.append(tempList)

        #
        headers = ['sigPath', 'filePath']
        csv_name = dir + ".csv"
        with open(self.csp_path + csv_name, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(filesList)
        #

        #return filesList


    def getFileList(self, path_txt):
        file_path_list = list()
        file_name_list = os.listdir(path_txt)

        file_name_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        # file_name_list.sort(key=lambda x: int(x.split('_')[1]))
        for i in range(len(file_name_list)):
            file_path = os.path.join(path_txt, file_name_list[i])
            file_path_list.append(file_path)
        return file_path_list

    # bei yong duoci chuanshu
    # def extragen_fileList(self, dir,lenth, fSig, fFile, basedir, sigStore, fileStore):
    #     filesList = self.filesDict[dir]
    #     for i in range(lenth):
    #         tempList = list()
    #         sigName = fSig[i].filename.split('/')
    #         fileName = fFile[i].filename.split('/')
    #         pathSig = os.path.join(basedir, 'UTC(demo2)', sigStore, sigName[0])
    #         pathFile = os.path.join(basedir, 'UTC(demo2)', fileStore, fileName[0])
    #         tempList.append(pathSig)
    #         tempList.append(pathFile)
    #         filesList.append(tempList)
    #
    #     return filesList


    def insert_blockList(self, optPos, lenth, fSig, fFile, basedir, sigStore, fileStore, dir):
        for i in range(lenth):
            tempList = list()
            sigName = fSig[i].filename.split('/')
            fileName = fFile[i].filename.split('/')
            pathSig = os.path.join(basedir, 'UTC(demo4)', sigStore, sigName[0])
            pathFile = os.path.join(basedir, 'UTC(demo4)', fileStore, fileName[0])
            tempList.append(pathSig)
            tempList.append(pathFile)

            #self.filesList.insert(int(optPos), tempList)
            filesList = self.filesDict[dir]
            assert isinstance(filesList, list)
            filesList.insert(int(optPos), tempList)

        return self.filesDict


    def delete_blockList(self, optPos, dir):
        filesList = self.filesDict[dir]
        assert isinstance(filesList, list)

        type = 2
        optPos = int(optPos)
        sigPath = filesList[optPos][0]
        filePath = filesList[optPos][1]
        os.remove(sigPath)
        os.remove(filePath)

        filesList.pop(optPos)

        return self.filesDict


    def update_blockList(self, optPos, lenth, fSig, fFile, basedir, sigStore, fileStore, dir):
        filesList = self.filesDict[dir]
        assert isinstance(filesList, list)

        optPos = int(optPos)
        #delete org file according to the path stored in list
        sigPath = filesList[optPos][0]
        filePath = filesList[optPos][1]
        os.remove(sigPath)
        os.remove(filePath)

        #recover the list
        for i in range(lenth):
            sigName = fSig[i].filename.split('/')
            fileName = fFile[i].filename.split('/')
            pathSig = os.path.join(basedir, 'UTC(demo4)', sigStore, sigName[0])
            pathFile = os.path.join(basedir, 'UTC(demo4)', fileStore, fileName[0])


            filesList[optPos][0] = pathSig
            filesList[optPos][1] = pathFile

        return self.filesDict


    def delete_flieList(self, basedir, dir):
        path = os.path.join(basedir, 'UTC(demo4)', dir)
        shutil.rmtree(path)
        self.filesDict.pop(dir)

        return self.filesDict


    def update_flieList(self, lenth, fSig, fFile, basedir, sigStore, fileStore, dir):
        self.delete_flieList(basedir, dir)
        filesList =  self.gen_fileList(lenth, fSig, fFile, basedir, sigStore, fileStore)
        self.filesDict[dir] = filesList

        return self.filesDict







if __name__ == '__main__':
    M = head.group.init(ZR, 0)
    T = head.group.init(G1, 1)
    print(M)
    print(type(M))
    print(T)
    print(type(T))
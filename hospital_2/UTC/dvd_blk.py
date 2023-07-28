# -*- coding: utf-8 -*-

# 按照大小分割文件
import csv
import json
import math
import os
import pickle
import time

import dlit
from dlit import DLIT
import pandas as pd

filePath = list()

# 将文件分块后写入新文件夹
def mk_SubFile(srcName, sub, buf):
    new = '/home/andy/UTCfiles/test/fileStore/'
    # new = os.getcwd() + '/test/fileStore/'
    [des_filename, extname] = os.path.splitext(srcName)
    if not os.path.exists(new):     #判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(new)
    filename = new + des_filename + '_' + str(sub) + extname
    filePath.append(filename)
    print('正在生成子文件: %s' % filename)
    with open(filename, 'wb') as fout:
        fout.write(buf)
        return sub + 1

'''
def list_txt(path, list=None):
    
    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist
'''


# 给指定用户文件分块
def split_By_size(fid, uid):#1
    filename = input("Enter your file name: ")
    while (not (os.path.exists(filename))):
        print("file:", filename, "doesn't exist. Please input again:")
        filename = input("Enter your file name: ")
        print("input file name is: ", filename)

    blocksize = input("Enter your block size(bytes)：")
    while (not blocksize.isdigit()):
        print("input size:", blocksize, "isn't digit.Please input again:")
        blocksize = input("Enter your block size(bytes)：")

    print("input block size is: ", blocksize, "bytes")

    blocksize = int(blocksize)
    size = os.path.getsize(filename)
    print("your filesize is", size, "bytes")
    blockNumber = math.ceil(size / blocksize)
    print("your file will be divided into", blockNumber, "blocks")
    #22222222222222222222222222222222222222
    #dt = dltgen(blockNumber, fid, uid)#3
    dt = dlt2csv(blockNumber,fid,uid)
    with open(filename, 'rb') as fin:
        buf = fin.read(blocksize)
        sub = 0
        while len(buf) > 0:
            sub = mk_SubFile(filename, sub, buf)
            buf = fin.read(blocksize)
    print("ok")
    return blocksize, blockNumber, size, dt, filePath

#111111111111111111111111111111111111111111111111(yuanban)
def dltgen(blockNumber, fid, uid):#2
    V = input('input version:')
    V = int(V)
    VT = list()
    for i in range(blockNumber):
        VT.append((V, time.time()))
    dlit = DLIT(fid, uid, VT)
    print("dlit info:", dlit.printDLIT())
    return dlit

#ba vt xieru csv wenjian
def dlt2csv(blockNumber, fid, uid):  # 2
    headers = ['V', 'T']
    save_path = r'/home/andy/UTCfiles/TPA/'
    fid = str(fid)
    uid = str(uid)
    csv_name = fid + uid + ".csv"
    V = input('input version:')
    V = int(V)
    VT = list()
    for i in range(blockNumber):
        VT.append((V, time.time()))
    with open(save_path+csv_name, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(VT)
    dlit = DLIT(fid, uid, VT)
    print(VT)
    print("dlit info:", dlit.printDLIT())
    return dlit

def test():
    save_path = r'/media/yorio/61B6-9D42/Data/'
    csv_name = "11.csv"
    result = os.popen('sed -n {}p {}'.format(3, save_path+csv_name)).read()
    
    print(result)

def test1():
    csp_path = r'/media/yorio/61B6-9D42/Data/CSP/'
    df = pd.read_csv("/media/yorio/61B6-9D42/Data/CSP/11.CSV")
    df.to_parquet("/media/yorio/61B6-9D42/Data/CSP/large11.parquet", compression=None)
    df = pd.read_parquet("/media/yorio/61B6-9D42/Data/CSP/large11.parquet", engine="fastparquet").loc[5]
    print(df)

def test2():
    df = pd.read_csv("/media/yorio/61B6-9D42/Data/TPA/11.CSV").head(-1)
    print(df)
if __name__ == '__main__':
    test2()
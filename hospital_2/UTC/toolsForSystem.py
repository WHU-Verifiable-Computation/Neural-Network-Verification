import csv
import random

import pandas as pd
# 给指定用户文件分块
import math
import os

from dvd_blk import dltgen, dlt2csv

filePath = list()

def mk_SubFile(srcName, sub, buf):
    new = '/home/yorio/backupUTC/test/fileStore/'
    # new = os.getcwd() + '/test/fileStore/'
    [des_filename, extname] = os.path.splitext(srcName)
    if not os.path.exists(new):     #判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(new)
    filename = new + des_filename + '_' + str(sub) + extname
    #filePath.append(filename)
    print('正在生成子文件: %s' % filename)
    with open(filename, 'wb') as fout:
        fout.write(buf)
        return sub + 1


def split_By_row(fid, uid):
    filename = input("Enter your .csv file name: ")
    while (not (os.path.exists(filename))):
        print("file:", filename, "doesn't exist. Please input again:")
        filename = input("Enter your file name: ")
        print("input file name is: ", filename)
    rowCount = input("Enter the number of the row per dividing：")
    while (not rowCount.isdigit()):
        print("input number:", rowCount, "isn't digit.Please input again:")
        rowCount = input("Enter your block size(bytes)：")

    print("input number is: ", rowCount, "pieces")

    rowCount = int(rowCount)
    df = pd.read_csv(filename)
    count = df.shape[0]
    print("your filesize is", count)
    blockNumber = math.ceil(count / rowCount)
    print("your file will be divided into", blockNumber, "blocks")
    dt = dlt2csv(blockNumber, fid, uid)

    with open(filename, 'rt') as f:
        dict = []
        reader = csv.reader(f)
        next(reader)
        #rowCount = 25
        count = 0
        sub = 0

        for row in reader:
            dict.append(row)
            count = count + 1
            if count % rowCount == 0 and len(dict) > 0:
                sub = mk_SubFile(filename,sub,dict)
                dict = []
    return rowCount, blockNumber, count, dt, filePath

def mk_SubFile(srcName, sub, dict):
    new = '/home/andy/UTCfiles/test/fileStore/'

    [des_filename, extname] = os.path.splitext(srcName)
    if not os.path.exists(new):     #判断当前路径是否存在，没有则创建new文件夹
        os.makedirs(new)
    filename = new + des_filename + '_' + str(sub) + extname
    filePath.append(filename)
    pd.DataFrame(dict).to_csv(filename, sep="\t", header=False, index=False)
    print('正在生成子文件: %s' % filename)
    return sub + 1


#xiugai an bilv d wenjiankuai
def mess_blocks(choiceRate,path):
    choiceRate = choiceRate / 100
    count = len(os.listdir(path))
    assert 0 < choiceRate <= 1

    sn = math.floor(count * choiceRate)
    result = os.listdir(path)
    random.shuffle(result)
    result = result[:sn]
    print(result)

    for i in result:
        filePath = path+i
        f = open(filePath, 'r+')
        f.write("123")



if __name__ == '__main__':
    mess_blocks(1,"/media/yorio/61B6-9D42/Data/CSP/11/fileStore/")
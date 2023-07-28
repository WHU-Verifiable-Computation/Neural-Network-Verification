import time

import pandas as pd

import UTC.head
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT, pair
import UTC.myToolBox
import pickle

class TPA:
    def __init__(self):
        self.group = UTC.head.group
        self.g = UTC.head.g
        #x is secret key, to do modify later

        self.verInfos = []

        self._x = UTC.head.userParam['x']
        self.v = self.g ** self._x
        self.u = UTC.head.userParam['u']

        self.fileName = 'shen' + UTC.head.userParam['fileName']
        self.fnameSig = self.group.hash(self.fileName, G1) ** self._x
        self.dlit = None



    def challenge(self, fid, uid, metdNum, CNum, bool=False, num=-1):
        #blockCount = 5
        ################################################blockCount = self.dlit.getBlockCount(fid, uid)#19
        save_path = r'/home/andy/UTCfiles/TPA/'
        csv_name = fid + uid + ".csv"
        results = pd.read_csv(save_path+csv_name)
        # print(results)
        blockCount = len(results)  # 19
        metdNum = int(metdNum)

        #2023/07/6
        if(bool == True):
            t0 = time.perf_counter()
            selectBlockSet = []
            selectBlockSet.append(num)

            chal = {i: self.group.random(ZR) for i in selectBlockSet}
            print(chal)
            t1 = time.perf_counter()
            print("spend time is:", (t1 - t0))

        if(bool == False):
            if metdNum == 1:
                CNum = int(CNum)
                assert CNum > 0
                assert CNum <= 100
                t0 = time.perf_counter()
                selectBlockSet = UTC.myToolBox.getUniqueRandomNumByRate(blockCount, CNum)
                chal = {i: self.group.random(ZR) for i in selectBlockSet}
                t1 = time.perf_counter()
                print("spend time is:", (t1 - t0))
            elif metdNum == 2:
                Times = input("input how many times that you want to challenge:")
                CNum = int(CNum)
                Times = int(Times)
                assert CNum > 0
                assert CNum <= blockCount
                t0 = time.perf_counter()
                selectBlockSet = UTC.myToolBox.getUniqueRandomNumList(blockCount, CNum, Times)
                chal = list()
                for k in range(len(selectBlockSet)):
                    challenge = {i: self.group.random(ZR) for i in selectBlockSet[k]}
                    chal.append(challenge)
                t1 = time.perf_counter()
                print("spend time is:", (t1 - t0))
                # print(chal)


            else:
                CNum = int(CNum)
                t0 = time.perf_counter()
                selectBlockSet = UTC.myToolBox.getUniqueRandomNum(blockCount, CNum)
                chal = {i: self.group.random(ZR) for i in selectBlockSet}

                #2023/07/16
                with open("/home/andy/code/sys_collection/hospital_2/UTC/challengeBlocks.txt","w",encoding='utf-8') as f:
                    for i in range(len(selectBlockSet)):
                        f.write(str(selectBlockSet[i])+'\n')

                print(chal)
                t1 = time.perf_counter()
                print("spend time is:", (t1 - t0))
            with open('/home/andy/code/sys_collection/hospital_2/UTC/selected_list.txt', 'w') as file:
                # 将列表数据转换为字符串，并写入文件
                file.write('\n'.join(str(item) for item in selectBlockSet))

        return chal

    def verify(self, chal, proof,uidfid):#20
        assert isinstance(proof, dict)


        tpa_path = r'/home/andy/UTCfiles/TPA/'
        tpa_name = uidfid + ".csv"
        tpa_path = tpa_path + tpa_name
        print(tpa_path)
        t0 = time.perf_counter()
        DI = self.group.init(GT, 1)
        for i in chal.keys():
            # hwi = self.group.hash(self.verInfos[int(i)], G1)
            ###################################################
            # V = self.dlit.getVerifyInfo(fid,uid,int(i))[0]#21
            # T = self.dlit.getVerifyInfo(fid,uid,int(i))[1]#22
            ###################################################
            data = pd.read_csv(tpa_path,float_precision='round_trip').loc[int(i)]
            V = data[0]
            T = data[1]
            VT = str(int(V)) + str(T)
            # print(V)
            # print(T)
            # print(VT)
            hwi = self.group.hash(VT, G1)
            # hwi = self.group.hash(self.dlit.getVerifyInfo[i], G1)
            DI *= self.group.pair_prod(hwi, self.v) ** chal[i]
        lhs = self.group.pair_prod(proof['T'], self.g)
        rhs = DI * self.group.pair_prod(self.u ** proof['M'], self.v)
        t1 = time.perf_counter()
        print("spend time is:", (t1 - t0))

        return lhs == rhs
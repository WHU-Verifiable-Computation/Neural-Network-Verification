from UTC import TPAServer


def checkEooro(fid=2, uid=2):
    f = open("/home/andy/code/sys_collection/hospital_2/UTC/challengeBlocks.txt","r")
    blockList = f.readlines()
    for i in range(len(blockList)):
        num = int(blockList[i])
        print(num)
        TPAServer.challenge(fid=str(fid), uid=str(uid), metdNum=0, CNum=1, bool=True, num=num)
        f = open("/home/andy/code/sys_collection/hospital_2/UTC/result.txt","r")
        onceBool = f.readline()
        print(str(num)+onceBool)
        if(onceBool == "False"):
            with open(r"/home/andy/code/sys_collection/hospital_2/UTC/checkResult.txt", "w", encoding='utf-8') as f_error:
                f_error.write(str(num)+'\n')

if __name__ == '__main__':
    checkEooro()
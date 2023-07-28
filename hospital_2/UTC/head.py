'''
this python file for sharing

the PairingGroup:group,
pulblic key:g,
CSP's secret key:sk
'''
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, GT, pair

import UTC.publicParam, UTC.userSecretData

testDatafilePath = '/home/yorio/backupUTC/UTC(demo2)/test/'   # ceshi wenjian dizhi
testDatafilePathDO = '/home/yorio/Desktop/test0/test0'
#testDatafilePathDO = '/home/yorio/backupUTC/UTC(demo2)/data.txt'
#testDatafilePathDO = '/home/yorio/Desktop/test0/1.jpg'

def getUserParam(k):
    userParam = {
        'x': group.deserialize(UTC.userSecretData.sks[k]),
        'u': group.deserialize(UTC.publicParam.us[k]),
        'fileName': UTC.userSecretData.sks[k].decode()
    }
    return userParam


group = PairingGroup('SS512')
g = group.deserialize(UTC.publicParam.g)
userParam = {
    'x' : group.deserialize(UTC.userSecretData.sks[0]),
    'u' : group.deserialize(UTC.publicParam.us[0]),
    'fileName' : UTC.userSecretData.sks[0].decode()
}

from enum import Enum, unique
@unique
class User2TPAMessageType(Enum):
    INSERTBLOCK = 1
    DELETEBLOCK = 2
    MODIFYBLOCK = 3
    DELETEFILE = 4
    MODIFYFILE = 5




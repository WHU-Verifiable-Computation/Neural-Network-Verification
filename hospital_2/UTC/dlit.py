class FileNode:
	def __init__(self, fid, uid, pre = None, next = None):
		self.fid = fid
		self.uid = uid
		self.pre = pre
		self.next = next
		self.blocks = None
		self.blockCount = 0
class BlockNode:
	def __init__(self, V, T, pre = None, next = None):
		self.V = V
		self.T = T
		self.pre = pre
		self.next = next


class DLIT:
	#上传一个文件初始化DLIT
	def __init__(self, fid, uid, VT):
		self.head = self.buildRow(fid, uid, VT)
		self.tail = self.head
		self.fileCount = 1

	def buildRow(self, fid, uid, VT):
		fileNode = FileNode(fid, uid)
		fileNode.blockCount = len(VT)
		pre = None
		for i, (v, t) in enumerate(VT):
			if i == 0:
				pre = BlockNode(v, t)
				fileNode.blocks = pre
				continue
			cur = BlockNode(v, t)
			pre.next = cur
			cur.pre = pre
			pre = cur

		return fileNode

	def findFileNode(self, fid, uid):
		cur = self.head
		pre = None
		while cur:
			if cur.fid == fid and cur.uid == uid:
				return pre, cur
			pre = cur
			cur = cur.next

		return None, None

	def insertFile(self, fid, uid, VT):
		#默认查到最后
		fileNode = self.buildRow(fid, uid, VT)
		fileNode.pre = self.tail
		self.tail.next = fileNode
		self.tail = fileNode
		self.fileCount = self.fileCount + 1

	def deleteFile(self, fid, uid):
		pre, fileNode = self.findFileNode(fid, uid)
		self.fileCount = self.fileCount - 1
		#要删除的为头节点
		if not pre:
			fileNode.next.pre = None
			self.head = fileNode.next

		#要删除的节点为尾节点
		elif not fileNode.next:
			pre.next = None
			self.tail = pre

		#中间节点	
		else: 
			pre.next = fileNode.next
			fileNode.next.pre = pre


	def modifyFile(self, fid, uid, VT):
		newFileNode = self.buildRow(fid, uid, VT)
		pre, fileNode = self.findFileNode(fid, uid)
		print(pre)
		#修改头节点
		if not pre:
			fileNode.next.pre = newFileNode
			newFileNode.next = fileNode.next
			self.head = newFileNode

		#修改尾节点
		elif not fileNode.next:
			pre.next = newFileNode
			newFileNode.pre = pre
			self.tail = newFileNode

		#中间节点
		else:
			nxt = fileNode.next	
			pre.next = newFileNode
			newFileNode.pre = pre
			newFileNode.next = nxt
			nxt.pre = newFileNode

	#根据论文，在第i个节点后面插入节点
	def insertBlock(self, fid, uid, i, V, T):
		i = int(i)
		pre, fileNode = self.findFileNode(fid, uid)
		blockNode = BlockNode(V, T)
		fileNode.blockCount = fileNode.blockCount + 1
		#在头节点前插
		if i == 0:
			fileNode.blocks.pre = blockNode
			blockNode.next = fileNode.blocks
			fileNode.blocks = blockNode
			return
		i = i - 1
		cur = fileNode.blocks
		while i:
			cur = cur.next
			i = i - 1

		#在尾节点后插
		if not cur.next:
			cur.next = blockNode
			return

		cur.next.pre = blockNode
		blockNode.next = cur.next
		cur.next = blockNode
		blockNode.pre = cur

	
	def deleteBlock(self, fid, uid, i):
		i = int(i)
		pre, fileNode = self.findFileNode(fid, uid)
		fileNode.blockCount = fileNode.blockCount - 1
		#删除头节点
		if i == 0:
			deleteNode = fileNode.blocks
			fileNode.blocks = deleteNode.next
			deleteNode.next.pre = None
			return

		i = i - 1
		cur = fileNode.blocks
		while i:
			cur = cur.next
			i = i - 1

		#删除尾节点
		if not cur.next:
			cur.pre.next = None
			return

		cur.next.pre = cur.pre
		cur.pre.next = cur.next



	def modifyBlock(self, fid, uid, i, V, T):
		i = int(i)
		pre, fileNode = self.findFileNode(fid, uid)
		cur = fileNode.blocks
		while i:
			cur = cur.next
			i = i - 1
		cur.V = V
		cur.T = T

	def printDLIT(self):
		cur = self.head
		while cur:
			blocks = cur.blocks
			while blocks:
				print(blocks.V, blocks.T)
				blocks = blocks.next
			cur = cur.next

	def getBlockCount(self, fid, uid):
		pre, fileNode = self.findFileNode(fid, uid)
		return fileNode.blockCount

	def getVerifyInfo(self, fid, uid, i):
		pre, fileNode = self.findFileNode(fid, uid)
		cur = fileNode.blocks
		while i:
			cur = cur.next
			i = i - 1
		return cur.V, cur.T

def main1():
	dlit = DLIT(1, 1, [(1,1), (1,2), (1,3)])
	dlit.printDLIT()
	#dlit.insertFile(2, 1, [(1,4),(1,5),(1,6)])
	# dlit.printDLIT()
	temp0 = dlit.findFileNode(1,1)
	temp1 = dlit.findFileNode(2,1)
	a = temp1[0]
	print(temp0)
	print(temp1)
	if a is None:
		print('123')
		return True
	# dlit.deleteFile(1, 1)
	# dlit.insertFile(3, 1, [(1,7),(1,8),(1,9)])
	# dlit.printDLIT()
#
# def main2():
# 	dlit = DLIT(1, 1, [(1,1), (1,2), (1,3)])
# 	#dlit.printDLIT()
# 	dlit.insertFile(2, 1, [(1,4),(1,5),(1,6)])
# 	#dlit.printDLIT()
# 	dlit.deleteFile(2, 1)
# 	dlit.insertFile(3, 1, [(1,7),(1,8),(1,9)])
# 	dlit.printDLIT()
#
def main3():
	dlit = DLIT(1, 1, [(1,1), (1,2), (1,3)])
	# dlit.insertFile(2, 1, [(1,4),(1,5),(1,6)])
	dlit.modifyFile(1, 1, [(2,7),(2,8),(2,9)])
	# dlit.insertFile(3, 1, [(1, 7), (1, 8), (1, 9)])
	dlit.printDLIT()

# def main4():
# 	dlit = DLIT(1, 1, [(1, 1), (1, 2), (1, 3)])
# 	#插头
# 	dlit.insertBlock(1, 1, 0, 0, 1, 4)
# 	#插尾
# 	dlit.insertBlock(1, 1, 4, 0, 1, 5)
# 	#插中间
# 	dlit.insertBlock(1, 1, 2, 0, 1, 6)
# 	dlit.printDLIT()

def main5():
	dlit = DLIT(1, 1, [(1, 1), (1, 2), (1, 3),(1, 4), (1, 5)])
	#删头
	dlit.deleteBlock(1, 1, 0)
	#删尾
	dlit.deleteBlock(1, 1, 4)
	#删中间
	dlit.deleteBlock(1, 1, 2)
	dlit.printDLIT()

def main6():
	dlit = DLIT(1, 1, [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)])
	dlit.modifyBlock(1, 1, 2, 2, 6)
	dlit.printDLIT()
	print(dlit.getBlockCount(1, 1))
	print(dlit.getVerifyInfo(1, 1, 2))

	dlit.insertBlock(1, 1, 0, 1, 4)

	print(dlit.getBlockCount(1, 1))

	dlit.deleteBlock(1, 1, 0)

	print(dlit.getBlockCount(1, 1))
	dlit.printDLIT()

if __name__ == '__main__':
	#删除文件头节点
    #main1()
    #删除文件尾节点
    #main2()
	#修改文件节点
	main3()
	

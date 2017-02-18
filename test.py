__author__ = 'Moon'
import sys

class ListNode(object):
    def __init__(self,x):
        self.val = x
        self.next = None


def generateLinkList(NodeList):
    if len(NodeList) == 0:
        return None
    else:
        list_nodes = map(lambda x:ListNode(x),NodeList)
        for i in range(len(NodeList)-1):
            list_nodes[i].next = list_nodes[i+1]
        return list_nodes[0]

def testLinkList(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def numSquares(n):
        """
        :type n: int
        :rtype: int
        """
        sqrt = 1
        count = 0
        sq_list = []
        while sqrt**2<=n:
            sq_list.append(sqrt**2)
            sqrt += 1
        nums = [4]*(n+1)
        for i in range(1,n+1):
            if i in sq_list:
                nums[i] = 1
            else:
                for item in sq_list:
                    if item<i:
                        nums[i] = min(nums[i],nums[i-item]+1)
                    else:
                        break
        return nums[-1]

print(numSquares(2))

def extract_text(file_path):
    file = open(file_path)
    lines = file.readlines()
    print(len(lines))
    count = 0
    error = 0
    for line in lines:
        try:
            text = yaml.load(line.strip())
        except:
            error += 1
            continue
        if text["extracted_text"]:
            count += 1
    print(count)
    print(error)
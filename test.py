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

def mergeList(head1,head2):
        if head1 == None or head2 == None:
            return head1 or head2
        dummy = ListNode(0)
        cur = dummy
        while head1 and head2:
            if head1.val<=head2.val:
                cur.next = head1
                head1 = head1.next
            else:
                cur.next = head2
                head2 = head2.next
            cur = cur.next
        cur.next = head1 or head2
        return dummy.next

def lengthLongestPath(input):
        """
        :type input: str
        :rtype: int
        """
        dir_parts = input.split("\n") #O(n) split each line as separate parts
        print(dir_parts)
        stack = []
        max_len = 0
        for part in dir_parts:
            num_of_tab = part.count("\t")
            while num_of_tab < len(stack): #If they equal, part is the last part in the file path chain and should be appended to the file path chain
                stack.pop()
            stack.append(len(part)-num_of_tab)
            print(stack)
            if "." in part:
                max_len = max(max_len,sum(stack)+len(stack)-1)
        return max_len
input = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
print(lengthLongestPath(input))
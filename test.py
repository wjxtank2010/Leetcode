__author__ = 'Moon'
import sys
def myPow(x, n): #50
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        abs_n = abs(n)
        power_list = [x]
        binary_list = [abs_n%2]
        power_value = x
        abs_n /= 2
        while abs_n>0:
            power_value *= power_value
            power_list.append(power_value)
            binary_list.append(abs_n%2)
            abs_n /= 2
        print(binary_list)
        print(power_list)
        result = reduce(lambda x,y:x*y,filter(lambda k:k != 0,map(lambda x,y:x*y,power_list,binary_list)))
        if n<0:
            return 1/result
        else:
            return result

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
a = [1,2,3,4,5,6,6,7]
b = []
head_a = generateLinkList(a)
head_b = generateLinkList(b)
#print(testLinkList(head_a),testLinkList(head_b))
result = mergeList(head_a,head_b)
print(testLinkList(result))


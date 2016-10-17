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

def trap(height):
    stack = []
    water = 0
    for index,value in enumerate(height):
        max_inter_height = 0
        print(stack)
        while stack:
            last = stack[-1]
            if last[1]<=value:
                water += (index-last[0]-1)*(last[1]-max_inter_height)
                ele = stack.pop()
                max_inter_height = ele[1]
            else:
                water += (index-last[0]-1)*(value-max_inter_height)
                #stack[-1] = (stack[-1][0],stack[-1][1],stack[-1][2]-value)
                break
        stack.append((index,value,value))
    return water

height = [2,1,0,2]
print(trap(height))


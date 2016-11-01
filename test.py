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

def numIslands(grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    #look around land to see if there is a neibor that has already been marked
                    if i-1>=0 and type(grid[i-1][j]) is int: #top
                        grid[i][j] = grid[i-1][j]
                    elif i+1<=len(grid)-1 and type(grid[i][j+1]) is int: #down
                        grid[i][j] = grid[i+1][j]
                    elif j-1>=0 and type(grid[i][j-1]) is int: #left
                        grid[i][j] = grid[i][j-1]
                    elif j+1<=len(grid[0])-1 and type(grid[i][j+1]) is int: #right
                        grid[i][j] = grid[i][j+1]
                    else:
                        count += 1
                        grid[i][j] = count
        print(grid)
        return count


#grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
grid = [["1"],["1"]]
print(numIslands(grid))


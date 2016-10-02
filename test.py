__author__ = 'Moon'

def combinationSum4(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        dp = [[[]]]+[[] for i in xrange(target)]
        for i in xrange(1,target+1):
            for number in nums:
                if number>i:
                    break
                else:
                    for item in dp[i-number]:
                        dp[i] += item+[number],
        return len(dp[target])

#print(combinationSum4([1,50],200))


def countComponents(n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        points = range(n)
        i = 0
        while points:
            source = points.pop()
            queue = [source]
            while queue:
                #print(queue,i)
                sub_source = queue.pop()
                print(edges,sub_source)
                for edge in edges[:]:
                    if edge[0] == sub_source:
                        queue.append(edge[1])
                        edges.remove(edge)
                    if edge[1] == sub_source:
                        queue.append(edge[0])
                        edges.remove(edge)
                if sub_source in points:
                    points.remove(sub_source)
            i = i+1
        return i

def generateParenthesis(n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = [["",0]]
        for j in range(2*n):
            for i in range(len(result)):
                string = result[i][0]
                difference = result[i][1]
                if difference>0:
                    if (len(string)+difference)/2 < n:
                        result[i][0] += "("
                        result[i][1] += 1
                        result.append([string+")",difference-1])
                    else:
                        result[i][0] += ")"
                        result[i][1] -= 1
                else:
                    result[i][0] += "("
                    result[i][1] += 1
        return [item[0] for item in result]
answer = ["(((())))","((()()))","((())())","((()))()","(()(()))","(()()())","(()())()","(())(())","(())()()","()((()))","()(()())","()(())()","()()(())","()()()()"]
print(set(generateParenthesis(4))-set(answer))


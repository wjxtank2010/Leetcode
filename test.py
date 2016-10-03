__author__ = 'Moon'
import sys
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

def maxSubArray(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        def subArray(nums):
            if len(nums) == 0:
                return -sys.maxint-1
            mid = len(nums)/2
            leftmax,rightmax = -sys.maxint - 1,-sys.maxint - 1
            sum = 0
            for i in range(mid,-1,-1):
                sum += nums[i]
                leftmax = max(leftmax,sum)
            sum = 0
            for j in range(mid,len(nums)):
                sum += nums[j]
                rightmax = max(rightmax,sum)
            return max(leftmax+rightmax-nums[mid],subArray(nums[0:mid]),subArray(nums[mid+1:]))
        return subArray(nums)

nums = [-2,-1,-3,4,-1,2,1,-5,-4]
print(maxSubArray(nums))

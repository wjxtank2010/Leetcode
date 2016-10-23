import itertools,collections
class ListNode(object):     #Problem 2
    def __init__(self, x):
        self.val = x
        self.next = None
def addTwoNumbers(self, l1, l2): #2
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        indicator = 0
        val = 0
        nodeArray = []
        while True:
           if l1 != None and l2 != None:
               indicator, val = divmod(l1.val+l2.val+indicator,10)
               node = ListNode(val)
               l1 = l1.next
               l2 = l2.next
               if len(nodeArray)>0:
                   nodeArray[-1].next = node
               nodeArray.append(node)
           elif l1 != None and l2 == None:
               indicator, val = divmod(l1.val+indicator,10)
               node = ListNode(val)
               l1 = l1.next
               if len(nodeArray)>0:
                   nodeArray[-1].next = node
               nodeArray.append(node)
           elif l1 == None and l2 != None:
               indicator, val = divmod(l2.val+indicator,10)
               node = ListNode(val)
               l2 = l2.next
               if len(nodeArray)>0:
                   nodeArray[-1].next = node
               nodeArray.append(node)
           else:
               if indicator == True:
                   node = ListNode(1)
                   nodeArray[-1].next = node
                   nodeArray.append(node)
               break
        if len(nodeArray) == 0:
             return None
        else:
             return nodeArray[0]

def findMedianSortedArrays(self, nums1, nums2):     #Problem 4
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        if len(nums1)<=2:
            return self.Basecase(nums1,nums2)
        elif len(nums2)<=2:
            return self.Basecase(nums2,nums1)
        else:
            median_1 = float(nums1[len(nums1)/2]+nums1[(len(nums1)-1)/2])/2
            median_2 = float(nums2[len(nums2)/2]+nums2[(len(nums2)-1)/2])/2
            num_of_discards = min((len(nums1)-1)/2,(len(nums2)-1)/2)
            if median_1 == median_2:
                return median_1
            elif median_1 > median_2:
                return self.findMedianSortedArrays(nums1[:-num_of_discards])

def findMedianSortedArray(self,num):
    if len(num) == 0:
        return None
    else:
        return float(num[len(num)/2]+num[(len(num)-1)/2])/2

def Basecase(self,nums1,nums2):
    if len(nums2) == 0:
        return self.findMedianSortedArray(nums1)
    else:
        for num in nums1:
            lower_bound = 0
            upper_bound = len(nums2)-1
            while lower_bound <= upper_bound:
                mid = (lower_bound+upper_bound)/2
                if num >= nums2[mid]:
                    lower_bound = mid+1
                else:
                    upper_bound = mid-1
            nums2.insert(lower_bound,num)
    return findMedianSortedArray(nums2)


def reverse(x):     #Problem 7
    if x == 0:
        return 0
    x_str = str(x)
    if x<0:
        x_str = str(x)[1:]
    first_none_zero = False
    result = ''
    for i in range(len(x_str)-1,-1,-1):
        if first_none_zero:
            result = result+x_str[i]
        else:
            if int(x_str) == 0:
                continue
            else:
                result = result+x_str[i]
                first_none_zero = True
    upper_bound = 1
    i = 1
    while i<32:
        upper_bound = upper_bound*2
        i = i+1
    if int(result)<upper_bound:
        if x<0:
            return -int(result)
        else:
            return int(result)
    else:
        return 0


def atoi(str):      #Problem 8
    if len(str) == 0:
            return 0
    i = 0
    while i<len(str) and str[i] == ' ': #Skip white spaces
        i = i+1
    if i >= len(str):   #String is full of white spaces
        return 0
    else:
        sign = 1
        if str[i] == '+':
            i = i+1
        elif str[i] == '-':
            i = i+1
            sign = -1
        elif str[i].isdigit() == False :
            return 0
        n = 0
        j = i
        while j<len(str) and str[j].isdigit():
            old_n = n
            n = n*10+int(str[j])
            if n*sign>2147483647:
                return 2147483647
            elif n*sign<-2147483648:
                return -2147483648
            j = j+1
        if j>i:
            return n*sign
        else:
            return 0


def isPalindrome(x):  #Problem 9
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x<0:
            return False
        i = 10
        j = 1
        while x/i>0:
            i = i*10
            j = j+1
        y = x
        k = 1
        e = 10
        i = i/10
        while (k<=j/2):
            left_number = y/(i)
            y = y-y/(i)*(i)
            right_number = (x-(x/e)*e)/(e/10)   #**
            if left_number == right_number:
                i = i/10
                e = e*10
                k = k+1
            else:
                return False
        return True

def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """
    if len(height) == 0:
        return 0
    else:
        fifo_queue = []
        result = 0
        i = 0
        while (i<len(height))>0:
            while len(fifo_queue)>0:
                if height[fifo_queue[-1]]<=height[i]:
                    if (i<len(height)-1):
                        fifo_queue.append(i)
                        break
                    else:
                        result = max(result,height[fifo_queue[-1]]*(i-fifo_queue[0])+1)
                        fifo_queue.pop(0)
                else:
                    result = max(result,height[fifo_queue[-1]]*(i-fifo_queue[-1]))
                    fifo_queue.pop(-1)
            if (len(fifo_queue) == 0):
                fifo_queue.append(i)
            i = i+1
    return max(result,height[-1])

def intToRoman(num):  #Problem 12
    """
    :type num: int
    :rtype: str
    """
    result = ""
    i = 0
    while (num%10 > 0 or num/10 > 0):
        result = single_digit_to_Roman(num-(num/10)*10,i)+result
        num = num/10
        i = i+1
    return result

def single_digit_to_Roman(digit,power):  #function for problem 12
    dic = {"1":"I","5":"V","10":"X","50":"L","100":"C","500":"D","1000":"M"}
    result = ""
    key = str(10**power)
    midlevel_key = str(5*10**power)
    uplevel_key = str(10**(power+1))
    if (digit <= 3):
        result += dic[key]*digit
    elif (digit == 4):
        result = result + dic[key] + dic[midlevel_key]
    elif (digit>=5 and digit<=8):
        result = result + dic[midlevel_key] + (digit-5)*dic[key]
    else:
        result = result + dic[key]+ dic[uplevel_key]
    return result

def romanToInt(s):    #Problem 13
        """
        :type s: str
        :rtype: int
        """
        dic = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        result = 0
        i = 0
        while i<len(s):
            if (i == len(s)-1):
                result += dic[s[i]]
            else:
                for j in range(i+1,len(s)):
                    if (dic[s[i]]>dic[s[j]]):
                        result = result + dic[s[i]]*(j-i)
                        i = j-1
                        break
                    elif (dic[s[i]] == dic[s[j]]):
                        if (j == len(s)-1):
                            result = result + dic[s[i]]*(j-i+1)
                            i = j
                        else:
                            continue
                    else:
                        result = result + dic[s[j]] - (j-i)*dic[s[i]]
                        i = j
                        break
            i = i+1
        return result

def longestCommonPrefix(strs):   #Problem 14
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ''
        min_str_length = len(strs[0])
        common_prefix = ''
        for i in range(0,len(strs)):
            min_str_length = min(min_str_length,len(strs[i]))
        print(min_str_length)
        for i in range(0,min_str_length):
            for j in range(0,len(strs)):
                if strs[j][i] != strs[0][i]:
                    return common_prefix
            common_prefix = common_prefix+strs[0][i]
        return common_prefix

def isAnagram(s, t):  #Problem 242
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    if len(s) != len(t):
        return False
    i = 0
    dic_s = {}
    dic_t = {}
    while i<len(s):
        if s[i] in dic_s:
            dic_s[s[i]] += 1
        else:
            dic_s[s[i]] = 1
        if t[i] in dic_t:
            dic_t[t[i]] += 1
        else:
            dic_t[t[i]] = 1
        i = i+1
    print(dic_s,dic_t)
    if dic_s == dic_t:
        return True
    else:
        return False

def sum_bit(a,b):  #Problem 371i
    if a == 0:
        return b
    if b == 0:
        return a
    while (b != 0):
        carry = a & b    #calculate those bits that a and b are both 1 so that there would be carry for these bits
        a = a ^ b        #do add operation for bits that are 0 and 1 for a and b repectively
        b = carry << 1  #calculate the actual carry, if carry is still greater than 0, do another xor in the next loop
    return a

def invertTree(self, root):  #Problem 226
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    self.invertSubTree(root)
    return root

def invertSubTree(self,subRoot):
    if subRoot != None:
        subRoot.left,subRoot.right = subRoot.right, subRoot.left
        self.invertSubTree(subRoot.left)
        self.invertSubTree(subRoot.right)

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

a = ListNode(0)
b = ListNode(0)
a.next = b

def deleteNode(node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    node = node.next

def isPowerOfThree(n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return False
        while n%3 == 0:
            if n == 3:
                return True
            else:
                n = n/3
        return False

def isUgly(num):  #problem 263
        """
        :type num: int
        :rtype: bool
        """
        if num == 0:
            return False
        if num == 1:
            return True
        while num%2 == 0 or num%3 == 0 or num%5 == 0:
            print(num)
            if num == 2 or num == 3 or num == 5:
                return True
            else:
                if num%2 == 0:
                    num = num/2
                elif num%3 == 0:
                    num = num/3
                elif num%5 == 0:
                    num = num/5
        return False
def fibo(n):
    a = (1/(5**(0.5)))*(((1+5**(0.5))/2)**(n+1) - ((1-5**(0.5))/2)**(n+1))
    print(a)

def reverseVowels(s):
    """
    :type s: str
    :rtype: str
    """
    vowels = ["a","e","i","o","u","A","E","I","O","U"]
    if len(s) == 0:
        return ""
    first_part = ""
    second_part = ""
    i = 0
    i_1 = 0
    j = len(s)-1
    j_1 = len(s)-1
    while i<j:
        if s[i] in vowels:
            if s[j] in vowels:
                first_part += s[i_1:i]+s[j]
                second_part = s[i]+s[j+1:j_1+1]+second_part
                i += 1
                j -= 1
                i_1 = i
                j_1 = j
            else:
                j -= 1
        else:
            i += 1
        #print(first_part+" "+second_part)
    if j_1+1<=len(s)-1:
        first_part += s[i_1:j_1+1]
    return first_part+second_part

print(str(bin(-2147483648)))

def permutation(nums):
    if len(nums) == 1:
        return [nums]
    result = []
    for i in range(len(nums)):
        for j in permutation(nums[0:i]+nums[i+1:]):
            # print(j)
            result.append([nums[i]]+j)
    return result

def merge(nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = 0
        j = 0
        while i<m+n:
            while j<n:
                if nums1[i]>=nums2[j]:
                    nums1.insert(i,nums2[j])
                    i += 1
                    j += 1
                else:
                    break
            i += 1
        print(nums1)

def reverseBits(n):
        """
        :type n: int
        :rtype: int
        """
        digit_arr = []
        while n/2 > 0 or n%2 > 0:
            digit_arr.append(n%2)
            n = n/2

def singleNumber(nums):  #Problem 136
        """
        :type nums: List[int]
        :rtype: int
        """
        return reduce(lambda x,y:x^y,nums)
def addtion(a,b):
    xor = a^b
    carry = (a&b)<<1
    while carry>0:
        tmp = xor
        xor = xor^carry
        carry = (tmp&carry)<<1
    return xor

class Solution(object): #problem 189
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #method 1
        # if len(nums) > 0:
        #     for i in range(k):
        #         tmp = nums.pop()
        #         nums.insert(0,tmp)

        #method 2
        length = len(nums)
        mod = k%length
        nums[:mod],nums[mod:] = nums[length-mod:],nums[:length-mod]

def deleteDuplicates(head): #problem 83
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dump = ListNode(0)
        dump.next = head
        pre = dump
        cur = head
        dic = {}
        while cur != None:
            if cur.val in dic: #Nodes to delete
                pre.next = cur.next
                cur = cur.next
            else:
                dic[cur.val] = 1
                pre = cur
                cur = cur.next
        return dump.next

def isPalindrome(head): #problem 234
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None:
            return True
        if head.next == None:
            return True
        list_length = 0
        var = head
        while var != None:
            list_length += 1
            var = var.next
        stop_index = list_length/2
        pre = None
        cur = head
        stop_node = None
        start_node = None
        for i in range(stop_index):
            if i == stop_index-1:
                if list_length%2 == 0:
                    start_node = cur.next
                else:
                    start_node = cur.next.next
                cur.next = pre
                stop_node = cur
            else:
                next = cur.next
                cur.next = pre
                pre = cur
                cur = next
        while stop_node != None and start_node != None:
            if stop_node.val == start_node.val:
                stop_node = stop_node.next
                start_node = start_node.next
            else:
                return False
        if stop_node == None and start_node == None:
            return True
        else:
            return False

def addBinary(a, b): #problem 67
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        binary_str_a = "0b"+a
        binary_str_b = "0b"+b
        int_a = int(a,2)
        int_b = int(b,2)
        return bin((int_a+int_b))[2:]

class NumArray(object): #problem 303
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        if len(nums) == 0:
            self.nums = []
        else:
            self.sumDic = {}
            self.sumDic[0] = nums[0]
            for i in range(1,len(nums)):
                self.sumDic[i] = self.sumDic[i-1] + nums[i]
            self.nums = nums

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        if len(self.nums) == 0:
            return 0
        return self.sumDic[j]-self.sumDic[i]+self.nums[i]

class MinStack(object): #155

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.q = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        curMin = self.getMin()
        if curMin == None or curMin>x:
            curMin = x
        self.q.append((curMin,x))
    def pop(self):
        """
        :rtype: void
        """
        self.q.pop()
    def top(self):
        """
        :rtype: int
        """
        return self.q[-1][1]

    def getMin(self):
        """
        :rtype: int
        """
        if len(self.q)==0:
            return None
        return self.q[-1][0]

class Solution(object): #278
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        start = 1
        end = n
        while start<end:
            mid = (start+end)/2
            print(mid)
            if isBadVersion(mid):
                end = mid
            else:
                start = mid+1
        return start

def convertToTitle(self, n): #168
        """
        :type n: int
        :rtype: str
        """
        result = ""
        while n%26 > 0 or n/26>0:
            if n%26 == 0:
                result = "Z"+result
                n = n/26-1
                continue
            else:
                result = chr(n%26+64)+result
            n = n/26
        return result

def compareVersion(self, version1, version2): #165
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        version1_digits = version1.split(".")
        version2_digits = version2.split(".")
        i = 0
        while i<min(len(version1_digits),len(version2_digits)):
            int_1 = int(version1_digits[i])
            int_2 = int(version2_digits[i])
            if int_1 == int_2 :
                i += 1
            elif int_1 > int_2:
                   return 1
            else:
                return -1
        if len(version1_digits) == len(version2_digits):
            return 0
        elif len(version1_digits)>len(version2_digits):
            for j in range(i,len(version1_digits)):
                if int(version1_digits[j])>0:
                    return 1
            return 0
        else:
            for j in range(i,len(version2_digits)):
                if int(version2_digits[j])>0:
                    return -1
            return 0

def productExceptSelf(self, nums): #238
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        output = [1]
        p = 1
        for i in range(len(nums)-1):
            p = p*nums[i]
            output.append(p)
        p = 1
        i = len(nums)-1
        while i>=0:
            output[i] *= p
            p *= nums[i]
            i -= 1
        return output

def topKFrequent(self, nums, k): #347
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        dic = {}
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        distinct_list = dic.items()
        distinct_list.sort(key= lambda k:k[1],reverse = True)
        return [distinct_list[i][0] for i in range(k)]

def maxProfit(self, prices): #122
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) == 0:
            return 0
        max_profit = 0
        min = prices[0]
        max = prices[0]
        for i in range(len(prices)):
            if prices[i]>=max:
                max = prices[i]
                if i == len(prices)-1:
                    max_profit += max-min
            elif prices[i]<max:
                max_profit += max-min
                min = prices[i]
                max = prices[i]
        return max_profit

def countNumbersWithUniqueDigits(self, n): #357
        """
        :type n: int
        :rtype: int
        """
        i = 1
        count = 1
        tmp = 9
        while i <= min(n,10):
            if i > 1:
               tmp *= (11-i)
            count += tmp
            i += 1
        return count

def integerBreak(self, n): #343
        """
        :type n: int
        :rtype: int
        """
        if n == 2:
            return 1
        if n == 3:
            return 2
        #any integer appears in the final result must be no more than 3,or it can be decomposed into better result. Therefore only 2 or 3 would appear in the final result. two 3 equals three 2 while 3*3>2*2*2. Therefore we need as many 3 as possible to create max product.
        number_of_3 = 0
        number_of_2 = 0
        if n%3 == 1:
            number_of_3 = (n-4)/3
            number_of_2 = 2
        else:
            number_of_3 = n/3
            number_of_2 = (n-(n/3)*3)/2
        return 3**number_of_3*2**number_of_2

def missingNumber(self, nums):#268
        """
        :type nums: List[int]
        :rtype: int
        """
        sum = len(nums)
        nums_sum = 0
        for i in range(len(nums)):
            sum += i
            nums_sum += nums[i]
        return sum-nums_sum

def bulbSwitch(self, n): #319
        """
        :type n: int
        :rtype: int
        """
        #only number that has odd factors would turn on in the end which are square numbers. Therefore we only need to find how mnay
        #square numbers are less than n
        return int(n**0.5)

a = ["bcedengp","jegidiicfohjimcccnkagmanbkkmbmlfabgammipaiepjnfi","condccpkmicalappldjbnlepdplggcmcnilkkinefgdmldegcjbimfaikfjldpoplcakdkglpnlnjkojhcglig","gnpddclieoneddfhojknjkkaehlkgegpfbnopjcnogcicokhlffd","edjnfgmoaningkmcfncodeganbmbhamoighbojdcjcdicipdobbcahil","hkjkfjkmaagcgpcimmljfghmkgekmkh","homffkkkipepcbmipn","ipkakmolkhecdddpdpcphafiaofgb","jeejnmdgokbmkmpdlkgiolkemenahkoidimoeidpagefpcokodcmjdjcbkaeaogmcbenhdcnegg","eapffhadafnnalkkobdbmpnnhfeg","opfknkinpknipgjhcjdgffjoippjcnfabdejn","fhnpojhnfiogkdgffgjc","nlhmgmigpbconl","jjlejejegngdljbgkkomfacjfjoemilaeofgdhiip","imagjifpapmgej","mjkdjbjmbhfbkbjnahlenhkdnnpplpenidahdkokeihmjjfojndiidlbkh","ihpnbcncpigllmpdekkjbdemobbdjemafeioiaoibgabakmmkklopoeppibdfhdlallcedldclmkhpmkkdg","plfpneanjljoebogkbfabdnkanodgljecaemcffnoicmdbgkkfma","afianbgmdff","bhobekifemheldcamckoccfdcnmdcmnfbodfldbhgoikplggehfokkolacadgoholcdbgcpgoanmmnmika","bnjklpaiklipbcbphhfjoibhhhljoppcfjdno","ickghheki","kacnagbficmhikbdeldcagg","laamledghhflcjjdfofdojkfbkl","aacafidfbpmimlclhcjignenchbdjcgfbdiegmohfloieajekfgcmamifbmkjnoaipcfmbejmkigaekcgemffemfcnkhhkkj","hlcdacefjodhllpbhkkokepmcdmmigclefoglahalejfiipjelmhkbhjblmbkgpimjbmplmgclfiifncckibpkjlnkbgcph","ohdadlaokhbhlgiepimmdmfegefpbkbcodnfebbcfakfijhnefg","lgeokomognpmfblbhekgnlpdohnpkoajionmjhhagenafmcdpgfiidlmnogigfipdpgjljlgakaoammekhmjhecnmghecdhljl","kmjkfdbibcnkinkagennlcfbajffdkgajloinjakombekjcbh","iofimmgddmjidfmmloofkdihlffaakapkhchmcclocdbandon","kpbmkbdfgobjebadjdhgkglakcbjpjkkpd","ghjahnmogdofblllfpdfbhohkmdingfmaofoflchbhpkjccffadnbd","lobkhmbgcjdpkemkijhddofpadcgabbfadkhmdmgippceaapeabhbemoejfhejhjncceckddmldpbdlmeljmcpbnaflolihhaojlp","nlcmhaaebniidhgammlkhpphglanihnmpchebbehfgjeegdkicbnnkgnaipdhiemigfgaoehedph","ecdnklamhjbkemhngkldlbbbkalepnlci","eokamphdgfobplgmnhcigcjalmbldgmfcgbppcicooneopcmkginofpcmgleeplobcd","fcmjlicbblmgpoedaglfpdmiajjnfjdlklfoapdigokimnjfnboecnlejeklcifhoomlnmlfimmjdpfieamjnmmcklcflmjd","dnfkclagfkofamkdijjhhklenidoahjeoieaddohalfdbkjgfeojdonnnpbpnkofhafohinchffihnkmcdbhnfano","jajcnlbijmpkhmninjpdjlaiabdpeinep","lpffenbmipjalpmocceaegabghkkbldmaaef","hfnjdgjcllcepegpgebbdiod","blofdnemnlghmhnklnagnjdhhncbligogkmcankckkicohicfocenpdfojnkkbfaihbfkceclmnj","blkojdljgkfbgoocagophckjhhaedplcecjhni","bfbmhobnbpjipgnohnnggkodlnhnpdnnjjodjjhjpoiblenkmmjfn","dakooapcaocdjjlohkhodebfljekmohmhaajabfmfpkmgenpcgejdmbpfhdmjcndidgnckejn","mcdplfipobaabeljedjjgffgibkjkfhcknlelcjploff","hfkkkgc","bckkahmfgpmllpjlegffdhmamighjeejfeibomlcflnf","hdlfkfnncgfggofkhbnnoogdhcjbmiecemdhblmaclcjjhfkd","hlleonkggomlodjiicahjpacmfmnjfhefghfjfjjhmollnmceckfl","fikabchalojpfe","eabebmiboflaoeilplfgbodlnpdjdiigcchpimdmgmimfdallmcig","piaaepcaceeiiejnliffickdbjnpbfggbdlcjcbggmbfegohjnnemkbjiaamcnplelnmlfjcab","olaganiohokddaakddildkphfmohabaneglhkapeeefkjdcb","obknlbdepjhofdgpjmbiefollhjdmkcgiembmcejbekalcdfpmdmpedblmodhkgmdnneheegkebc","fbelnnohpboamjkbddhlbcdllnlkgllkafiionmipaafpeibf","cdmdiekekdfagofmfnffnclcfjajllhojglodmpiaffddknehallhoojhjmgjifeofmagdjfengmpohechphnkghdjhmnldh","bfjegjnfdeipaogfeifnnpgpbblfdc","poenmckimbpifflnjcnejemhjdifmibniogckajponoamnoldfefdoglmhlldicekiecnmomp","nbchhcmpbialmi","jdebllagcaidnkmbbadfmnjjhiondmphngmmdkjmbmfecfdhhggmllmnjicdokbiekbpmkebhhcjjbolcppnomldpckdfij","dbbjijglkgaomhphbfpaejiiknadcoaopo","gdcakpbpcpeebncijnldoigfnm","njkhafdm","hahieknapoaidphimlidackdnhhdcnlfhhahjfdkacbdfdfeeogeklcfmmlfeiamjfpmbaiolgcgoncandlcabahfmdk","iagkkbchfijjnjggeiipikldigbinhdoppcaldcbognoakidpppchhdknedhlfmgmgmkombekejdigiimoefmahmahhhgdcm","jkpkfmpgfknpicjkodpjhmfhiikgdifapoldfojbahpbjbgdhiookababepckambnc","afcaeklcpjhjfampafogehdpiianokcpclohciajpmbaplgfpkjnkallpkoghglaccnfhnllkinigchddaak","hmkbgninnnmebgaghblo","npfpcpfekkcmb","dchjgnnbbdhllkdihdcdpkjdjmhnamjpfjalnambcpfmlahocemncmhoockmgedlegcfjnomdo","mh","kbbibjkm","fiopkjgdgpodnkpiekggelkhblbkeaoekdfpomgmcijfchankfmffiacafkcnbpp","pcemmimjiipcaimpipmpachpjlnpmdoneiag","ceoeipfliflfjnmpkngcoidgjicpapfaogiegfbhgpflnbjbgpencgaopmepkmfnkkjblphcgabbincgajfgelapmhjloefpogkpp","clockpjegiikjgkgjofgdgbjejknn","kjkjklbhfdmjmhnpmjoehniigbmofjmbmjkfmcahpgabamahpmimgicbfoemfomgdifoajmlnjbpbdfcilcidibhejkkbbhe","egkjneklne","aohokpnbnjchkkfipbfpjcbmnpebdklngnmfjaj","fhcnhadlelgajengoocmcjfmglpanhiciffmikjjgiboblmcbfomkdlkhpeoidfgemhccpogimlnpdemfia","jbemhdfkcoldoanpmedafkbpmnnfmhdmlncciojcmecoglahffmgkhgpofeegoebdnlfpebgapbflcma","hfhbipcidfdfgbhhmahnfgpliagmolhhfbmcjfgnhkdkale","nllcogabcbbfggkknkjimmbdlfinamdifcbgoffhekkhpfflpfnleccapgcepgankpnblofmphplpnomfdkdbcdhejp","dgpjgjnagacjfhlkhnmpfbhcplfehoolafmpieoh","mcjhhgajmclkoiobmgaaellbekaadlkopgbkobmmipeollfcjgo","lhnlcgljdpcbcjmncpiojli","obcddnfdiogpchomf","kjhkbnpcphmpnpilkhbjmeohodcoonanlmniifpiiiipcegagnhpbpfadgo","cjbakbbjaikfjnbimkeifcoinhhkcogokknajeagmfkcldfijhhfapnnjnljnecknlgdmapnlhffdndahoj","hdpflnimifempccgjolbdbdalfaapfijpmjnkaedkgmbpanmilleeahmnpiipna","cpjkhblnleglklhhdolggpgfmgahhilhbjgdhegkkeppiibnif","cldhiemiaoeaahi","cnjbbeedhddcjcelhcmfnmfcmpo","piekkbklhnpokeeiidbhbcjfgmddkejhdhfcenabobmjleemneikgkjokmcikiemgjhdmjcni","lknlpnlnnck","hgooknomoahjpebbcnidnagdfghooihkhjmeadhbojodhfkgkcbgmcbfkeammfhnlpbmoggfcgbcmfllalal","jljohilkaekocjcfdmpjgcpbpolfobdpmojcpmphlpnedaheholbbpdnnfoibpacmjmgglboodfoldicnjmodcipa","colgmamhgjeckgiiehibkipdl","afiljflhejeanlimdhcicfebapebcfefchdadfchfklcdgbkmagpkofpnphhhpbbpebkabdppocfbag"]

def maxProduct(self, words): #318
        """
        :type words: List[str]
        :rtype: int
        """
        d = {}
        for word in words:
            bit_field = 0
            for char in word:
                bit_field |= 1<<(ord(char)-97)
            if bit_field in d:
                d[bit_field] = max(d[bit_field],len(word))
            else:
                d[bit_field] = len(word)
        # d = {sum(1 << (ord(c) - 97) for c in set(w)): len(w) for w in words}
        return max([d[k] * d[K] for k, K in itertools.combinations(d.keys(), 2) if not K & k] or [0])

class Solution(object):  #366
    def findLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        dic = {}
        self.markNodes(root,dic)
        return [k[1] for k in sorted(dic.items(),key= lambda x: x[0])]


    def markNodes(self,root,dic):
        if root == None:
            return 0
        height = max(self.markNodes(root.left,dic),self.markNodes(root.right,dic))+1
        if height in dic:
            dic[height].append(root.val)
        else:
            dic[height] = [root.val]
        return height

def wiggleSort(self, nums): #280  O(nlogn)
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return
        nums.sort()
        i = 1
        while i<len(nums)-1:
            nums[i],nums[i+1] = nums[i+1],nums[i]
            i += 2

def wiggleSort(self, nums):#280, O(n) runtime much slower  ???
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in range(1,len(nums)):
            if (i%2 == (nums[i-1]>nums[i])):
                nums[i-1],nums[i] = nums[i],nums[i-1]

import random
class Solution(object): #382
    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.head = head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        if self.head.next == None:
            return self.head.val
        i = 2
        cur = self.head.next
        result = self.head.val
        while cur != None:
            tmp = random.random()
            if tmp <= 1.0/i:
                result = cur.val
            cur = cur.next
            i += 1
        return result

def plusOne(self, head): #369
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        dummy = ListNode(0)
        dummy.next = head
        cur = dummy
        add_list = []
        while cur.next:
            if cur.next.val == 9:
                add_list.append(cur)
            else:
                add_list = []
            cur = cur.next
        if cur.val == 9:
            cur.val = 0
            for item in add_list:
                item.val = (item.val+1)%10
        else:
            cur.val = cur.val+1
        if dummy.val == 1:
            return dummy
        else:
            return dummy.next

def getModifiedArray(self, length, updates): #370
        """
        :type length: int
        :type updates: List[List[int]]
        :rtype: List[int]
        """
        result = [0]*length
        for update in updates:
            result[update[0]] += update[2]
            if update[1]+1<length:
                result[update[1]+1] += -update[2]
        sum = 0
        for i in range(len(result)):
            sum += result[i]
            result[i] = sum
        return result

def twoSum(self, numbers, target): #167
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        i = 0
        j = len(numbers)-1
        while i<j:
            tmp = numbers[i]+numbers[j]
            if target>tmp:
                i += 1
            elif target<tmp:
                j -= 1
            else:
                return [i+1,j+1]

class Solution(object): #384
    def __init__(self, nums):
        """

        :type nums: List[int]
        :type size: int
        """
        self.original_nums = nums[:]
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        self.nums = self.original_nums[:]
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        random.shuffle(self.nums)
        return self.nums


def multiply(self, A, B): #311
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: List[List[int]]
        """
        result = []
        for i in range(len(A)):
            result.append([0]*len(B[0]))
        dic_A = {}
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] != 0:
                    dic_A[(i,j)] = A[i][j]
        print(dic_A)
        dic_B = {}
        for i in range(len(B)):
            for j in range(len(B[0])):
                if B[i][j] != 0:
                    dic_B[(i,j)] = B[i][j]
        print(dic_B)
        for key_A,value_A in dic_A.items():
            for key_B,value_B in dic_B.items():
                if key_A[1] == key_B[0]:
                    result[key_A[0]][key_B[1]] = result[key_A[0]][key_B[1]] + value_A*value_B
                    #print((key_A,key_B,value_A*value_B,result[key_A[0]][key_B[1]]))
        # Better Solution
        # result = []
        # for i in range(len(A)):
        #     result.append([0]*len(B[0]))
        # dic_B = {}
        # for i,row in enumerate(B):
        #     dic_B[i] = {}
        #     if any(row):
        #         for j,ele in enumerate(row):
        #             if ele != 0:
        #                 dic_B[i][j] = B[i][j]
        # for i,row_A in enumerate(A):
        #     if any(row_A):
        #         for k,eleA in enumerate(row_A):
        #             if eleA:
        #                 for j,eleB in dic_B[k].items():
        #                    result[i][j] += eleA*eleB
        return result

class Solution(object): #339
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        return self.subDepthSum(nestedList,1)


    def subDepthSum(self,nestedList,depth):
        result = 0
        for item in nestedList:
            if item.isInteger():
                result += item.getInteger()*depth
            else:
                result += self.subDepthSum(item.getList(), depth+1)
        return result

class Solution(object): #364
    def depthSumInverse(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        dic = {}
        self.subdepthSumInverse(nestedList,dic,1)
        if dic:
            maxDepth = max(dic.keys())
            result = 0
            for key,value in dic.items():
                result += (maxDepth-key+1)*value
            return result
        else:
            return 0

    def subdepthSumInverse(self,nestedList,dic,depth):
        for item in nestedList:
            if item.isInteger():
                if depth in dic:
                    dic[depth] += item.getInteger()
                else:
                    dic[depth] = item.getInteger()
            else:
                self.subdepthSumInverse(item.getList(),dic,depth+1)

def shortestDistance(self, words, word1, word2): #243
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        distance = len(words)
        word1_index = -1
        word2_index = -1
        for index,word in enumerate(words):
            if word == word1:
                word1_index = index
                if word2_index>= 0:
                    distance = min(distance,abs(word1_index-word2_index))
            elif word == word2:
                word2_index = index
                if word1_index>=0:
                    distance = min(distance,abs(word1_index-word2_index))
        return distance

class WordDistance(object): #244
    def __init__(self, words):
        """
        initialize your data structure here.
        :type words: List[str]
        """
        self.dic = {}
        for index,word in enumerate(words):
            if word in self.dic:
                self.dic[word].append(index)
            else:
                self.dic[word] = [index]

    def shortest(self, word1, word2):
        """
        Adds a word into the data structure.
        :type word1: str
        :type word2: str
        :rtype: int
        """
        word1_index = self.dic[word1]
        word2_index = self.dic[word2]
        distance = abs(word1_index[0]-word2_index[0])
        for index_1 in word1_index:
            for index_2 in word2_index:
                distance = min(distance,abs(index_1-index_2))
        return distance

def shortestWordDistance(self, words, word1, word2): #245
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        distance = len(words)
        if word1 == word2:
            word_index = -1
            for index,word in enumerate(words):
                if word == word1:
                    if word_index>=0:
                        distance = min(distance,index-word_index)
                        word_index = index
                    else:
                        word_index = index
        else:
            word1_index = -1
            word2_index = -1
            for index,word in enumerate(words):
                if word == word1:
                    word1_index = index
                    if word2_index>= 0:
                        distance = min(distance,abs(word1_index-word2_index))
                elif word == word2:
                    word2_index = index
                    if word1_index>=0:
                        distance = min(distance,abs(word1_index-word2_index))
        return distance


class ZigzagIterator(object): #281

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self.v1 = v1
        self.v2 = v2
        self.turn = 1

    def next(self):
        """
        :rtype: int
        """
        if self.turn == 1:
            self.turn = 2
            if self.v1:
                return self.v1.pop(0)
            else:
                return self.v2.pop(0)
        else:
            self.turn = 1
            if self.v2:
                return self.v2.pop(0)
            else:
                return self.v1.pop(0)

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.v1 or self.v2:
            return True
        else:
            return False

def minCost(self, costs): #256
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        if costs:
            result = [0,0,0]
            for index,row in enumerate(costs):
                result = [min(result[1],result[2])+row[0],min(result[0],result[2])+row[1],min(result[0],result[1])+row[2]]
            return min(result)
        else:
            return 0

def generatePossibleNextMoves(self, s): #293
    """
    :type s: str
    :rtype: List[str]
    """
    result = []
    index = -1
    for i,char in enumerate(s):
        if char == "+":
            if index>=0:
                if i-index == 1:
                    result.append(s[:index]+"--"+s[i+1:])
                index = i
            else:
                index = i
    return result

def oddEvenList(self, head): #328
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        if head.next == None:
            return head
        last_odd = head
        first_even = head.next
        last_even = head.next
        cur = first_even.next
        while cur:
            last_odd.next = cur
            tmp = None
            if cur.next:
                tmp = cur.next.next
            last_even.next = cur.next
            last_even = cur.next
            cur.next = first_even
            last_odd = cur
            cur = tmp
        return head

def sortTransformedArray(self, nums, a, b, c): #360
        """
        :type nums: List[int]
        :type a: int
        :type b: int
        :type c: int
        :rtype: List[int]
        """
        if len(nums) == 0:
            return []
        if a == 0:
            result = []
            for num in nums:
                result.append(b*num+c)
            if b<0:
                result = result[::-1]
            return result
        axis = -float(b)/(2*a)
        axis_index = 0
        axis_value = abs(nums[0]-axis)
        for index,num in enumerate(nums): #find the point that is nearest to the symmetric axis
            distance = abs(num-axis)
            if distance<axis_value:
                axis_index = index
                axis_value = distance
        print(axis,axis_value)
        result = []
        i = axis_index
        j = i + 1
        #print(i,j)
        while i >= 0 or j <= len(nums)-1:
            if i<0: #all nums before axis have been appended to the result
                result.append(a*nums[j]**2+b*nums[j]+c)
                j += 1
            elif j>= len(nums): #all nums after axis have been appended to the result
                result.append(a*nums[i]**2+b*nums[i]+c)
                i -= 1
            else:
                if abs(nums[i]-axis)<=abs(nums[j]-axis):
                    result.append(a*nums[i]**2+b*nums[i]+c)
                    i -= 1
                else:
                    result.append(a*nums[j]**2+b*nums[j]+c)
                    j += 1
            print(i,j)
        if a<0:
            result = result[::-1]
        return result

def maxSubArrayLen(self, nums, k): #325
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sum = 0
        dic = {0:0}
        res = 0
        for index,value in enumerate(nums):
            sum += value
            if sum-k in dic:
                res = max(res,index-dic[sum-k]+1)
            if sum not in dic:
                dic[sum] = index+1
        return res

def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        if len(heights) == 0:
            return 0
        stack = []
        result = 0
        for index,value in enumerate(heights):
            pop_element = None
            while stack:
                if stack[-1][1]>value:
                    pop_element = stack.pop()
                    result = max(result,pop_element[1]*(index-pop_element[0]))
                else:
                    break
            if pop_element:
                stack.append((pop_element[0],value))
            else:
                stack.append((index,value))
        #print(stack)
        while stack:
            pop_element = stack.pop()
            result = max(result,(len(heights)-pop_element[0])*pop_element[1])
        return result

class Solution_378(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        n = len(matrix)
        L, R = matrix[0][0], matrix[n-1][n-1]
        while L < R:
            mid = L + ((R-L) >> 1)
            temp = self.search_lower_than_mid(matrix, n, mid)
            if temp < k:
                L = mid + 1
            else:
                R = mid
        return L

    def search_lower_than_mid(self, matrix, n, x):
        i = 0
        j = n - 1
        cnt = 0
        while i < n and j >= 0:
            if matrix[i][j] <= x:
                i += 1
                cnt += j + 1
            else:
                j -= 1
        return cnt


def upsideDownBinaryTree(self, root): #156
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        last_node = None
        last_right = None
        cur = root
        while cur:
            next = cur.left
            tmp = cur.right
            cur.right = last_node
            cur.left = last_right
            last_node = cur
            last_right = tmp
            cur = next
        return last_node

class Solution_230(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        for val in self.inOrderTraversal(root):
            if k == 1:
                return val
            else:
                k -= 1

    def inOrderTraversal(self,root):
        if root:
            for node in self.inOrderTraversal(root.left):
                yield node
            yield root.val
            for node in self.inOrderTraversal(root.right):
                yield node

class Solution_39(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        if candidates:
            candidates.sort()
            result = self.combine(candidates,len(candidates)-1,target)
            return [m for m in result if m]
        else:
            return []

    def combine(self,arr,end,target):
        if target == 0:
            return [[]]
        else:
            if end < 0:
                return [None]
            if target<arr[0]:
                return [None]
            ele = arr[end]
            result = []
            for i in range(target/ele+1):
                tmp = self.combine(arr,end-1,target-i*ele)
                for j in range(len(tmp)):
                    if tmp[j] != None:
                        tmp[j] += [ele]*i
                result += tmp
            return result

def combinationSum(candidates, target): #39  dynamic programming by creating a list first record all the methods built up from 1 to target
    candidates.sort()
    dp = [[[]]] + [[] for i in xrange(target)]
    for i in xrange(1, target + 1):
        for number in candidates:
            if number > i: break
            for L in dp[i - number]:
                if not L or number >= L[-1]: dp[i] += L + [number],  #list addtion with "," indicates add an element into the list
    return dp[target]

def combinationSum2(self, candidates, target): #40
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        dic = collections.Counter(candidates)
        candidates.sort()
        dp = [[[]]] + [[] for x in xrange(target)]
        for i in xrange(1,target+1):
            for number in candidates:
                if number>i:
                    break
                for item in dp[i-number]:
                    if not item or (number>= item[-1] and dic[number]>item.count(number)):
                        if (item+[number]) not in dp[i]:
                            dp[i] += item+[number],
        return dp[target]

def combinationSum3(self, k, n): #41
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        number = [i for i in xrange(1,10)]
        result = []
        for item in itertools.combinations(number,k):
            if sum(item) == n:
                result.append(item)
        return result

class Solution_204(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        isPrime_list = [1]*(n-2)
        i = 2
        while i*i<n:
            if isPrime_list[i-2]:
                if self.isPrime(i):
                    increment = i*i
                    while increment<n:
                        isPrime_list[increment-2] = 0
                        increment += i
            i += 1
        return sum(isPrime_list)

    def isPrime(self,n):
        i = 2
        while i*i <n:
            if n%i == 0:
                return False
            i += 1
        return True

def strStr(self, haystack, needle): #28
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        #KMP table
        partial_match_table = [0 for i in xrange(len(needle))]
        a = 0
        for index in xrange(1,len(needle)):
            while a>0 and needle[a] != needle[index]:
                a = partial_match_table[a-1]
            if needle[a] == needle[index]:
                a += 1
            partial_match_table[index] = a


        index = 0
        while index<=len(haystack)-len(needle):
            i = 0
            print(index)
            isMatch = True
            while i < len(needle):
                if haystack[index+i] != needle[i]:
                    isMatch = False
                    if i == 0:
                        index += 1
                    else:
                        index += i - partial_match_table[i-1]
                    break
                i += 1
            if isMatch:
                return index
        return -1

def shortestPalindrome(self, s): #214
        """
        :type s: str
        :rtype: str
        """
        combine_s = s+"#"+s[::-1]
        kmp_table = [0 for i in range(len(combine_s))]
        print(combine_s)
        a = 0
        for i in xrange(1,len(combine_s)):
            while a>0 and combine_s[a] != combine_s[i]:
                a = kmp_table[a-1]
            if combine_s[a] == combine_s[i]:
                a += 1
            kmp_table[i] = a
        print(kmp_table)
        return s[:kmp_table[-1]-1:-1]+s

def isSubsequence(self, s, t): #392
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        index = -1
        for char in s:
            tmp = t[index+1:].find(char)
            if tmp == -1:
                return False
            index += tmp+1
        return True

        #Solution by StefanPochmann
        t = iter(t)
        return all(c in t for c in s)

def countComponents(self, n, edges): #323
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
                sub_source = queue.pop()
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

        #union find with path compression
        # points = range(n)
        # def find(v):
        #     if v != points[v]:
        #         points[v] = find(points[v])
        #     return points[v]
        # for e,g in edges:
        #     points[find(e)] = find(g)
        # return len(set(map(find,points)))

class Logger(object): #359

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic = {}

    def shouldPrintMessage(self, timestamp, message):
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        if message in self.dic:
            if timestamp-self.dic[message] <10:
                return False
            else:
                self.dic[message] = timestamp
                return True
        else:
            self.dic[message] = timestamp
            return True


class MovingAverage(object): #346

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.window = []
        self.sum = 0.0
        self.size = size
    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        if len(self.window)<self.size:
            self.window.append(val)
            self.sum += val
        else:
            rm_ele = self.window.pop(0)
            self.window.append(val)
            self.sum += val - rm_ele
        return self.sum/len(self.window)


def canPermutePalindrome(self, s): #266
        """
        :type s: str
        :rtype: bool
        """
        dic = {}
        for char in s:
            if char in dic:
                dic[char] ^= 1
            else:
                dic[char] = 1
        return sum(dic.values())<=1

def findTheDifference(self, s, t): #389
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        bit = 0
        for char in s+t:
            bit ^= 1 << (ord(char)-97)
        index = 0
        while bit>1:
            bit >>= 1
            index += 1
        return chr(97+index)

        #transform string into a series of bit is not necessary
        # bit = 0
        # for char in s+t:
        #     bit ^= ord(char)
        # return chr(bit)


def canConstruct(self, ransomNote, magazine): #383
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        ransom_list = [0]*26
        magazine_list = [0]*26
        for char in ransomNote:
            ransom_list[ord(char)-97] += 1
        for char in magazine:
            magazine_list[ord(char)-97] += 1
        for i in range(26):
            if magazine_list[i]-ransom_list[i]<0:
                return False
        return True

        #using python collections module
        #return not collections.Counter(ransomNote) - collections.Counter(magazine)

def canAttendMeetings(self, intervals):#252
        """
        :type intervals: List[Interval]
        :rtype: bool
        """
        if len(intervals) == 0 :
            return True
        intervals.sort(key = lambda k:k.start)
        tmp = -1
        for item in intervals:
            if item.start>=tmp:
                tmp = item.end
            else:
                return False
        return True

def firstUniqChar(self, s): #387
        """
        :type s: str
        :rtype: int
        """
        dic = {}
        single_char = []
        remaining_index = 26
        for index,char in enumerate(s):
            if char not in dic:
                dic[char] = index
                single_char.append(char)
            else:
                if char in single_char:
                    single_char.remove(char)
        if single_char:
            return dic[single_char[0]]
        else:
            return -1

        #Just enumerate all the cases
        #lowercase = "abcdefghijklmnopqrstuvwxyz"
        #return min([s.find(char) for char in lowercase if s.count(char)==1] or [-1])


def readBinaryWatch(self, num):#401
        """
        :type num: int
        :rtype: List[str]
        """
        result = []
        for i in range(num+1):
            for hour in itertools.combinations([8,4,2,1],i):
                hour_value = sum(hour)
                if hour_value<12:
                    for minute in itertools.combinations([32,16,8,4,2,1],num-i):
                        minute_value = sum(minute)
                        if minute_value<10:
                            result.append(str(hour_value)+":"+"0"+str(minute_value))
                        elif minute_value<60:
                            result.append(str(hour_value)+":"+str(minute_value))
        return result

def isStrobogrammatic(self, num): #246
        """
        :type num: str
        :rtype: bool
        """
        dic = {"1":"1","6":"9","8":"8","9":"6","0":"0"}
        first_half_rev = ""
        for char in num[:len(num)/2]:
            if char in dic:
                first_half_rev += dic[char]
            else:
                return False
        print(first_half_rev)
        print(num[:len(num)-len(num)/2-1:-1])
        if first_half_rev == num[:len(num)-len(num)/2-1:-1]:
            if len(num)%2 == 0:
                return True
            else:
                tmp = num[len(num)/2]
                if tmp == "1" or tmp == "8" or tmp == "0" :
                    return True
                else:
                    return False
        else:
            return False

def isStrobogrammatic(self, num): #246
        """
        :type num: str
        :rtype: bool
        """
        dic = {"1":"1","6":"9","8":"8","9":"6","0":"0"}
        for i in range(len(num)/2+1):
            if num[i] not in dic:
                return False
            if dic[num[i]] != num[~i]:
                return False
        return True

def closestValue(self, root, target): #270
        """
        :type root: TreeNode
        :type target: float
        :rtype: int
        """
        tmp = root
        result = root.val
        while tmp:
            if abs(tmp.val-target)<abs(result-target):
                result = tmp.val
            if tmp.val>target:
                tmp = tmp.left
            elif tmp.val<target:
                tmp = tmp.right
            else:
                return tmp.val
        return result

def groupStrings(self, strings): #249
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        result = []
        type_list = []
        for string in strings:
            string_pattern = []
            for i in range(1,len(string)):
                string_pattern.append((ord(string[i])-ord(string[i-1]))%26)
            if string_pattern in type_list:
                index = type_list.index(string_pattern)
                result[index].append(string)
            else:
                type_list.append(string_pattern)
                result.append([string])
        return result

def numWays(self, n, k): #276
    """
    :type n: int
    :type k: int
    :rtype: int
    """
    if n == 0:
        return 0
    if n == 1:
        return k
    first = 1
    second = k
    for i in range(n-2):
        #There are two different cases. For every plaint strategy for the n-1 fence, there are k-1 different choices for the nth position => f(n-1)*(k-1)
        #The other case is plainting the nth fence same color with (n-1)th, in that way the (n-1) and (n-2) can not have the same color. There are f(n-2)*(k-1)
        #different strategy for that (n-1)th position different (n-2)th position.
        #Therefore in total: (k-1)(f(n-1)+f(n-2))
        first,second = second,(k-1)*(first+second)
    return k*second

def findNthDigit(self, n): #400
        """
        :type n: int
        :rtype: int
        """
        index = 0
        sum = 0
        while sum < n:
            index += 1
            sum += 10**(index-1)*9*index
        sum -= 10**(index-1)*9*index
        mod,remainder = divmod(n-sum, index)
        print((mod,remainder))
        if remainder == 0:
            return int(str(10**(index-1)+mod-1)[-1])
        else:
            return int(str(10**(index-1)+mod)[remainder-1])

def generateParenthesis(n): #22
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

def rob(self, nums): #198
        """
        :type nums: List[int]
        :rtype: int
        """
        first,second = 0,0
        for item in nums:
            first,second = second,max(first+item,second)
        return second

def rob(self, nums): #213
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        #first element included
        first,second = nums[0],nums[0]
        for ele in nums[2:-1]:
            first,second = second,max(first+ele,second)
        #first element not included
        first_1,second_1 = 0,0
        for ele in nums[1:]:
            first_1,second_1 = second_1,max(first_1+ele,second_1)
        return max(second,second_1)

class House_Robber_III_Solution(object): #337
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return max(self.subRob(root))

    def subRob(self,root):
        if root == None:
            return (0,0)
        left = self.subRob(root.left)
        right = self.subRob(root.right)
        #The first value denotes max money containing root, while the second does not.
        #Max containing root must contain root plus max money without root from the left node and right node
        #Max without root containing max result from left and max result from right
        return (root.val+left[1]+right[1],max(left)+max(right))

def maxRotateFunction(self, A): #396
        """
        :type A: List[int]
        :rtype: int
        """
        base = sum(A)
        pre = 0
        for index,value in enumerate(A):
            pre += index*value
        result = pre
        for i in range(1,len(A)):
            cur = pre+base-A[-i]*len(A)
            pre,result = cur,max(cur,result)
        return result

def integerReplacement(self, n): #397
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n>1:
            if n%2 == 0:
                n >>= 1
            else:
                if n == 3:
                    count += 2
                    break
                if n & 3 == 3: # the last two bits are 11
                    n += 1
                else:
                    n -= 1
            count += 1
        return count

class TwoSum(object): #170    iterate over dic with TLE while list passes

    def __init__(self):
        """
        initialize your data structure here
        """
        self.dic = {}
        self.num = []

    def add(self, number):
        """
        Add the number to an internal data structure.
        :rtype: nothing
        """
        if number in self.dic:
            self.dic[number] += 1
        else:
            self.dic[number] = 1
            self.num.append(number)


    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        for key in self.num:
            if value-key in self.dic:
                if key == value-key:
                    if self.dic[key] >= 2:
                        return True
                else:
                    return True
        return False


def read(self, buf, n): #157  Still does not understand this problem: why do we need to create the empty buffer
        idx = 0
        while n > 0:
            # read file to buf4
            buf4 = [""]*4
            l = read4(buf4)
            # if no more char in file, return
            if not l:
                return idx
            # write buf4 into buf directly
            for i in range(min(l, n)):
                buf[idx] = buf4[i]
                idx += 1
                n -= 1
        return idx

class ValidWordAbbr(object): #288
    def __init__(self, dictionary):
        """
        initialize your data structure here.
        :type dictionary: List[str]
        """
        self.dic = {}
        for string in dictionary:
            if len(string) <= 2:
                if string not in self.dic:
                    self.dic[string] = (string,1)
            else:
                abbre = string[0]+str(len(string)-2)+string[-1]
                if abbre not in self.dic:
                    self.dic[abbre] = (string,1)
                else:
                    self.dic[abbre] = (string,2)

    def isUnique(self, word):
        """
        check if a word is unique.
        :type word: str
        :rtype: bool
        """
        abbre = ""
        if len(word)<=2:
            abbre = word
        else:
            abbre = word[0]+str(len(word)-2)+word[-1]
        if abbre in self.dic:
            if self.dic[abbre][0] == word and self.dic[abbre][1] == 1:
                return True
            else:
                return False
        else:
            return True


def singleNumber(self, nums): #137
        """
        :type nums: List[int]
        :rtype: int
        """
        #we build a counter for this, since there are up to three of the same num, two bits are enough
        # a  b  c  a  b
        # 0  0  0  0  0
        # 0  1  0  0  1
        # 1  0  0  1  0
        # 0  0  1  0  1
        # 0  1  1  1  0
        # 1  0  1  0  0
        # a = a&~b~c+~a&b&c
        # b = ~a&b&~c+~a&~b&c
        a = 0
        b = 0
        for c in nums:
            tmp_a = (a&~b&~c)|(~a&b&c)
            b = (~a&b&~c)|(~a&~b&c)
            a = tmp_a
        return a|b

def reverseWords(self, s): #151
        """
        :type s: str
        :rtype: str
        """
        return " ".join(s.split()[::-1])

def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        if len(points) <= 2:
            return len(points)
        slope_dic = {}
        for i in xrange(len(points)):
            for j in xrange(i+1,len(points)):
                if points[i].x == points[j].x:
                    if points[i].y != points[j].y:
                        if ("infi",points[i].x) not in slope_dic:
                            slope_dic[("infi",points[i].x)] = [i,j]
                        else:
                            if j not in slope_dic[("infi",points[i].x)]:
                                slope_dic[("infi",points[i].x)].append(j)
                    else:
                        if ("infi",points[i].x,points[i].y) not in slope_dic:
                            slope_dic[("infi",points[i].x,points[i].y)] = [i,j]
                        else:
                           if j not in slope_dic[("infi",points[i].x,points[i].y)]:
                                slope_dic[("infi",points[i].x,points[i].y)].append(j)
                else:
                    slope = float((points[j].y-points[i].y))/(points[j].x-points[i].x)
                    intersection = 0
                    if slope == 0:
                        intersection = points[j].y
                    else:
                        intersection = points[i].x-points[i].y/slope
                    if (slope,intersection) in slope_dic:
                        if i not in slope_dic[(slope,intersection)]:
                            slope_dic[(slope,intersection)].append(i)
                        if j not in slope_dic[(slope,intersection)]:
                            slope_dic[(slope,intersection)].append(j)
                    else:
                        slope_dic[(slope,intersection)] = [i,j]
        if slope_dic:
            return max(map(len,slope_dic.values()))
        else:
            return 0

def maxSubArray(self, nums): #53
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        else:
            result = nums[0]
            range_sum = nums[0]
            for num in nums[1:]:
                if num>0:
                    if result>0:
                        range_sum += num
                        result = max(range_sum,result)
                    else:
                        result = num
                        range_sum = num
                else:
                    if result<0:
                        result = max(result,num)
                    else:
                        if range_sum+num>0:
                            range_sum += num
                        else:
                            range_sum = 0
            return result

        #Accumulative sum == subtration between two accmulative sum starting from 0
        # if len(nums) == 0:
        #     return 0
        # min = nums[0]
        # sum = nums[0]
        # result = nums[0]
        # for item in nums[1:]:
        #     sum += item
        #     if sum<min:
        #         min = sum
        #     else:
        #         result = max(result,sum-min)
        # return max(result,max(nums))

        #Divide and Conquer runtime O(nlog(n))
        #if len(nums) == 0:
        #     return 0
        # def subArray(nums):
        #     if len(nums) == 0:
        #         return -sys.maxint-1
        #     mid = len(nums)/2
        #     leftmax,rightmax = -sys.maxint - 1,-sys.maxint - 1
        #     sum = 0
        #     for i in range(mid,-1,-1):
        #         sum += nums[i]
        #         leftmax = max(leftmax,sum)
        #     sum = 0
        #     for j in range(mid,len(nums)):
        #         sum += nums[j]
        #         rightmax = max(rightmax,sum)
        #     return max(leftmax+rightmax-nums[mid],subArray(nums[0:mid]),subArray(nums[mid+1:]))
        # return subArray(nums)

def addTwoNumbers(self, l1, l2): #2
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        cur = dummy
        carry = 0
        while l1 and l2:
            node = ListNode(0)
            carry,node.val = divmod(l1.val+l2.val+carry,10)
            cur.next = node
            cur = node
            l1,l2 = l1.next,l2.next
        tmp = None
        if l1:
            tmp = l1
        elif l2:
            tmp = l2
        cur.next = tmp
        while tmp:
            carry,tmp.val = divmod(tmp.val+carry,10)
            cur,tmp = tmp,tmp.next
        if carry:
            cur.next = ListNode(1)
        return dummy.next

def myPow(self, x, n): #50
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        abs_n = abs(n)
        power_list = [x]
        binary_list = [n%2]
        power_value = x
        abs_n /= 2
        while abs_n>0:
            power_value *= power_value
            if power_value>=sys.maxint:
                if n>0:
                    return sys.maxint
                else:
                    return 0
            power_list.append(power_value)
            binary_list.append(abs_n%2)
            abs_n /= 2
        result = reduce(lambda x,y:x*y,filter(lambda k:k != 0,map(lambda x,y:x*y,power_list,binary_list)))
        if n<0:
            return 1/result
        else:
            return result

        #make use of the reverse property and record each power is not necessary.
        # if n == 0:
        #     return 1
        # if n<0:
        #     n = -n
        #     x = 1/x
        # result = 1
        # while n>0:
        #     if n%2 == 1:
        #         result *= x
        #     x *= x
        #     n /= 2
        # return result

def wordBreak(self, s, wordDict): #139
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        char_list = [0]*len(s)
        for i in range(len(s)):
            for word in wordDict:
                if s[i-len(word)+1:i+1] == word and ((i-len(word) == -1) or char_list[i-len(word)]):
                    char_list[i] = 1
        print(char_list)
        return char_list[-1]>0

def findMin(self, nums): #153
        """
        :type nums: List[int]
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start<end:
            if nums[start]<=nums[end]:
                return nums[start]
            mid = (start+end)/2
            if nums[mid]>=nums[start]:
                start = mid+1
            else:
                end = mid
        return nums[start]

def maxArea(self, height): #11
        """
        :type height: List[int]
        :rtype: int
        """
        start = 0
        end = len(height)-1
        result = 0
        while start<end:
            start_value,end_value = height[start],height[end]
            if start_value<=end_value:
                result = max(start_value*(end-start),result)
                start += 1
            else:
                result = max(end_value*(end-start),result)
                end -= 1
        return result

class Solution(object):  #148
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        slow = dummy
        fast = dummy
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        head2 = self.sortList(slow.next)
        slow.next = None
        head1 = self.sortList(head)
        return self.mergeList(head1,head2)



    def mergeList(self,head1,head2):
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


class ListNode(): #146
    def __init__(self,key,value):
        self.key = key
        self.val = value
        self.pre = None
        self.next = None

class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dic = {}
        self.dummy = ListNode("head",0)
        self.tail = ListNode("tail",0)
        self.dummy.next,self.tail.pre = self.tail,self.dummy

    def get(self, key):
        """
        :rtype: int
        """
        if key in self.dic:
            node = self.dic[key]
            node.pre.next,node.next.pre = node.next,node.pre
            self.dummy.next.pre,node.next = node,self.dummy.next
            self.dummy.next,node.pre = node, self.dummy
            return node.val
        else:
            return -1


    def set(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: nothing
        """
        if key in self.dic:
            node = self.dic[key]
            node.val = value
            node.pre.next,node.next.pre = node.next,node.pre
            self.dummy.next.pre,node.next = node,self.dummy.next
            self.dummy.next,node.pre = node, self.dummy
        else:
            node = ListNode(key,value)
            if len(self.dic)<self.capacity:
                self.dummy.next.pre,node.next = node,self.dummy.next
                self.dummy.next,node.pre = node,self.dummy
            else:
                node_to_del = self.tail.pre
                del self.dic[node_to_del.key]
                if node_to_del != self.dummy:
                    node_to_del.pre.next,self.tail.pre = self.tail,node_to_del.pre
                    node_to_del = None
                    self.dummy.next.pre,node.next = node,self.dummy.next
                    self.dummy.next,node.pre = node,self.dummy
            self.dic[key] = node


def trap(self, height): #42 Stack
    """
    :type height: List[int]
    :rtype: int
    """
    stack = []
    water = 0
    for index,value in enumerate(height):
        max_inter_height = 0
        while stack:
            last = stack[-1]
            if last[1]<=value:
                water += (index-last[0]-1)*(last[1]-max_inter_height)
                ele = stack.pop()
                max_inter_height = ele[1]
            else:
                water += (index-last[0]-1)*(value-max_inter_height)
                break
        stack.append((index,value))
    return water
    #O(1) space
    # left,right = 0,len(height)-1
    # water = 0
    # maxleft,maxright = 0,0
    # while left<=right:
    #     if height[left]<=height[right]:
    #         if height[left] >= maxleft:
    #             maxleft = height[left]
    #         else:
    #             water += maxleft-height[left]
    #         left += 1
    #     else:
    #         if height[right] >= maxright:
    #             maxright = height[right]
    #         else:
    #              water += maxright-height[right]
    #         right -= 1
    # return water

def search(self, nums, target): #33
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums)-1
        while start<end:
            mid = (start+end)/2
            #It's an incresing array, just use the normal way to find a element
            if nums[start]<nums[end]:
                if nums[mid]<target:
                    start = mid + 1
                elif nums[mid]>target:
                    end = mid
                else:
                    return mid
            #It's still a rotated array,keep narrowing down
            else:
                #The max num is closer to end
                if nums[mid]>=nums[start]:
                    if target>nums[mid]:
                        start = mid+1
                    elif target<nums[mid]:
                        if target>nums[end]:
                            end = mid
                        elif target<nums[end]:
                            start = mid+1
                        else:
                            return end
                    else:
                        return mid
                #nums[mid]<end, indicating the max num is closer to start
                else:
                    if target>nums[mid]:
                        if target>nums[end]:
                            end = mid
                        elif target<nums[end]:
                            start = mid+1
                        else:
                            return end
                    elif target<nums[mid]:
                        end = mid
                    else:
                        return mid
        if nums[start] == target:
            return start
        else:
            return -1
>>>>>>> 394751d5c11a4a34b0fabd23e27c3e052b3978b8

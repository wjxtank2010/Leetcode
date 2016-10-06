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

print(myPow(-13.626083,3))


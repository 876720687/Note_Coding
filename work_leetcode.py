# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:52:43 2021

@author: 北湾

两数之和

"""
def twoSum_1(nums, target):
    lens = len(nums)
    j=-1
    for i in range(lens):
        # 这个if条件成立的话则说明确实是有这个数就在这个里面，那么使用index() 返回的就是查找对象的索引位置
        if (target - nums[i]) in nums:
            # 统计相减后的值在list 中出现的次数
            # 这个if说明找到的是它本身，题目设定输出只有一个答案，而且i和j不能相等
            
            if (nums.count(target - nums[i]) == 1 & (target - nums[i] == nums[i]) ):
                continue
            
            else:
                j = nums.index(target - nums[i], i+1)
                break
    if j > 0:
        return [i,j]
    else:
        return []

def twoSum_2(nums, target):
    lens = len(nums)
    j = -1
    for i in range(1,lens):
        
        temp = nums[:i]
        if (target - nums[i]) in temp:
           j = temp.index(target -nums[i])
           break
    if j >= 0:
        return [j, i]
        
def twoSum_3(nums, target):
    hashmap = {}
    for ind, num in enumerate(nums):
        hashmap[num] = ind
    for i, num in enumerate(nums):
        j = hashmap.get(target - num)
        if j is not None and i!=j:
            return [i, j]

if __name__ == '__main__':
    nums = [2,7,11,15]
    target = 9
    print(twoSum_1(nums,target))
    print(twoSum_2(nums, target))
    print(twoSum_3(nums, target))
 
'''
第二道题

'''

def addTwoNumbers(self, l1, l2):
    return
        test = l1.reverse()+l2.reverse()
    
    return (
                  (
                      ( (l1.reverse().join('')) ) +
                      ( +l2.reverse().join('') )
                  )
            ).split('').reverse()



# var addTwoNumbers = function(l1, l2) {
#     return String((((+l1.reverse().join(''))) + (+l2.reverse().join('')))).split('').reverse()
# };



if __name__ == '__main__':
    l1 = [2,4,3]
    l2 = [5,6,4]
    addTwoNumbers(l1, l2)
    print((l1.reverse()).extend(l2.reverse()))
    
print(l1.reverse())
list(l1.reverse())+list(l2.reverse())
a.extend(b)



# 像这种加点的都是内置方法！没有返回值
# 

list1 = ['physics', 'Biology', 'chemistry', 'maths']

a = list(list1.reverse())

b = list(reversed(list1))


'''
第三道题
'''

def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:max_len = cur_len
            lookup.add(s[i])
        return max_len

if __name__ == '__main__':
    s = "abcabcbb
    lengthOfLongestSubstring(s)
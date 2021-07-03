# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:49:51 2021

@author: 北湾
"""
'''
b1001

'''

n = int(input())
count = 0
while n!= 1:
    if n%2 == 1:
        n =(3*n+1)/2
    else:
        n = n/2
    count += 1
print(count)


'''
b1002
这个map的作用？
对list使用sum()会怎么样？
1 输入可以用map函数把每个字符转化为int类型，map返回一个map object，不过下面代码中用不用list转化都一样。
2 最后无空格。也可以用字符串的restrip方法，以后例子可能会用到。
'''

n = input()
x = list(map(int, n))
y =sum(x)
dt = {
      '0':'ling',
      '1':'yi',
      '2':'er',
      '3':'san',
      '4':'si',
      '5':'wu',
      '6':'liu',
      '7':'qi',
      '8':'ba',
      '9':'jiu',
      }
s = str(y)
for i in range(len(s)):
    if i != len(s)-1:
        print(dt[s[i]],end=' ')
    else:
        print(dt[s[i]],end='')


'''
b1003
没看懂

'''


'''
b1004
学习sorted函数的用法，顺带可以看一下使用Operator模块的高级排序
https://www.cnblogs.com/monsteryang/p/6938779.html

input()这个函数导致的无法同时粘贴多个样例的问题如何解决？

'''

def by_score(s):
    return s[2]

n = int(input())
student = []
while n:
    n -= 1
    name, idx, score = input().split()
    score = int(score)
    student.append((name, idx, score))

s_sorted = sorted(student, key=by_score)
print(s_sorted[-1][0], s_sorted[-1][1])
print(s_sorted[0][0], s_sorted[0][1])






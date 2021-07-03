def sift(li ,low ,high): # 实现了建堆的过程,大根堆
    """
    li：列表
    low：堆根节点的位置
    high：堆最后一个元素的位置
    """
    i = low
    j = i*2+1
    tmp = li[low]
    while j <= high:
        if j+1 <= high and li[j+1] > li[j]: # 右孩子存在并且比较大
            j =j + 1
        if li[j] > tmp:
            li[i] = li[j]
            i = j
            j = 2*i+1
        else:
            li[i] = tmp
            break
    else:
        li[i] = tmp

# def heap_sort(li):
#     n = len(len)
#     for i in range((n-2)//2, -1, -1):
#         sift(li, i, n-1)
#     for i in range(n-1, -1 -1):
#         li[0], li[i] = li[i], li[0]
#         sift(li, 0, i-1)

def heap_sort(li):
    n = len(li)
    for i in range((n-2)//2, -1, -1):
        sift(li, i ,n-1)
    # 建堆完成
    print(li)

li = [i for i in range(100)]
import random
random.shuffle(li)
print(li)
heap_sort(li)

"""
堆排序拥有内置模块 heapq
"""
import heapq
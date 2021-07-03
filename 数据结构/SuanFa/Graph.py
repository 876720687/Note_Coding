'''

图算法的实现
https://blog.csdn.net/qq_39422642/article/details/79473289


'''


class DenseGraph:
    def __init__(self, n, directed=False):
        self.n = n  # number of vertex
        self.m = 0  # number of edge
        self.directed = directed
        self.matrix = [[0 for i in range(n)] for i in range(n)]  # 用这样的方式定义了一个二维矩阵

    def __str__(self):
        for line in self.matrix:
            print(str(line))
        return ''

    def getNumberOfEdge(self):
        return self.m

    def getNumberOfVertex(self):
        return self.n

    def hasEdge(self, v, w):
        if 0 <= v <= self.n and 0 <= w <= self.n:
            return self.matrix[v][w]
        else:
            raise Exception("vertex not in the Graph")

    def addEdge(self, v, w):
        if 0 <= v <= self.n and 0 <= w <= self.n:
            if self.hasEdge(v, w):
                return
            self.matrix[v][w] = 1
            if self.directed is False:
                self.matrix[w][v] = 1
            self.m += 1
        else:
            raise Exception("vertex not in the Graph")


import regex as re


def buildGraphFromFile(aGraph, filePath):
    graphList = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            graphList.append([int(x) for x in re.split(r'\s+', line.strip())])
    for i in range(len(graphList)):
        aGraph.addEdge(graphList[i][0], graphList[i][1])


g1 = DenseGraph(13)  # 必须填入正确的结点个数。。。我真的觉得邻接矩阵不好用
buildGraphFromFile(g1, 'E:/CODE/ProgrammingProgect/pythoncoding/demo1/数据结构/testG1.txt')
print(g1)

from random import random, choice
import networkx as nx
import matplotlib.pyplot as plt


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


G = nx.Graph()

points = [(random(), random()) for _ in range(8)]

for p1, p2 in zip(points[:-1], points[1:]):
    G.add_edge(p1, p2, weight=dist(p1, p2))

for _ in range(8):
    p1, p2 = choice(points), choice(points)
    G.add_edge(p1, p2, weight=dist(p1, p2))

nx.draw(G)
plt.savefig('asd.png')
plt.show()
"""
一个使用 ConvexHull 及其属性的简单示例。 你可以适应
此处展示的技巧可帮助解决这次的问题。

请特别注意当您使用 ConvexHull 解决
二维坐标问题时，由此产生的结果“hull”是一个平面内的多边形。
这就是你需要的任务！

如果您要为其提供 3-D 坐标，则“hull”代表更多
复杂的几何实体。

"""

from scipy.spatial import ConvexHull


def summarize_hull_properties(ch: ConvexHull, max_items=20) -> None:
    """	打印 ConvexHull 对象的一些有用属性。
    :param ch: a scipy.spatial.ConvexHull
    :param max_items: 允许截断输出结果，当有很多点，顶点之类的时候
    """
    print("area: ", ch.area)
    print("points:\n", ch.points[:max_items])
    print("vertices: ", ch.vertices[:max_items])
    print("simplices:\n", ch.simplices[:max_items])
    print()
    return

'''这里我定义了两个具有相同角点的正方形，除了
box2 增加了一个额外的点。'''
box1 = [[0, 0],
        [1, 0],
        [1, 1],
        [0, 1]]

box2 = [[0, 0],
        [1, 0],
        [1, 1],
        [0.5, 0.5],  # this point is in the center of the square
        [0, 1]]

xs1 = list(np.random.randint(30,40,100))
ys1 = list(np.random.randint(20,30,100))
zs1 = list(np.random.randint(10,20,100))
box3=[[xs1],[ys1],[zs1]]



ch1 = ConvexHull(points=box1, incremental=True)
print("ch1:")
summarize_hull_properties(ch1)

print('''\n\n与上面相比，下面的hull包含一个额外的点在 0.5, 0.5. 我们可以通过这些方式判断它在hull内部而不是在其边界上
这些:
A) 添加该点并没有增加面积。
B) 该点不包括在顶点中。
C) 因此，该点也不是任何单纯形的端点。
''')

ch2 = ConvexHull(points=box2, incremental=True)
print("ch2:")
summarize_hull_properties(ch2)

print('\n接下来，我们在之前的 ConvexHulls 上添加一些外部点，并注意到情况发生了变化:')
ch1.add_points([[1.99, 0], [0, 2.0]])
print("ch1 is now:")
summarize_hull_properties(ch1)

ch2.add_points([[2.0, 0], [0, 2.0001]])
print("ch2 is now:")
summarize_hull_properties(ch2)

ch3 = ConvexHull(points=box3, incremental=True)
summarize_hull_properties(ch3)
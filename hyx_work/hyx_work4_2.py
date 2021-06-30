# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:35:13 2021

@author: admin
"""

from datetime import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import logging


class Gantt(object):
    '''
    简单地的线条渲染器
    '''

    # 红黄绿散色图
    RdYlGr = ['#d73027', '#f46d43', '#fdae61',
              '#fee08b', '#ffffbf', '#d9ef8b',
              '#a6d96a', '#66bd63', '#1a9850']

    POS_START = 1.0
    POS_STEP = 0.5

    def __init__(self, tasks):
        self._fig = plt.figure()
        self._ax = self._fig.add_axes([0.1, 0.1, .75, .5])

        self.tasks = tasks[::-1]

    def _format_date(self, date_string):
        '''
        将*date_string*的字符串表示格式设置为*matplotlib日期*实例。
        '''
        try:
            date = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        except ValueError as err:
            logging.error("String '{0}' can not be converted to datetime object: {1}"
                  .format(date_string, err))
            sys.exit(-1)
        mpl_date = mdates.date2num(date)
        return mpl_date

    def _plot_bars(self):
        '''
        处理每个任务并将barh添加到当前的self
        '''
        i = 0
        for task in self.tasks:
            start = self._format_date(task['start'])
            end = self._format_date(task['end'])
            bottom = (i * Gantt.POS_STEP) + Gantt.POS_START
            width = end - start
            self._ax.barh(bottom, width, left=start, height=0.3,
                          align='center', label=task['label'],
                          color = Gantt.RdYlGr[i])
            i += 1

    def _configure_yaxis(self):
        '''y axis'''
        task_labels = [t['label'] for t in self.tasks]
        pos = self._positions(len(task_labels))
        ylocs = self._ax.set_yticks(pos)
        ylabels = self._ax.set_yticklabels(task_labels)
        plt.setp(ylabels, size='medium')

    def _configure_xaxis(self):
        ''''x axis'''
        # make x axis date axis
        self._ax.xaxis_date()

        # format date to ticks on every 7 days
        rule = mdates.rrulewrapper(mdates.DAILY, interval=7)
        loc = mdates.RRuleLocator(rule)
        formatter = mdates.DateFormatter("%d %b")

        self._ax.xaxis.set_major_locator(loc)
        self._ax.xaxis.set_major_formatter(formatter)
        xlabels = self._ax.get_xticklabels()
        plt.setp(xlabels, rotation=30, fontsize=9)

    def _configure_figure(self):
        self._configure_xaxis()
        self._configure_yaxis()

        self._ax.grid(True, color='gray')
        self._set_legend()
        self._fig.autofmt_xdate()

    def _set_legend(self):
        '''
        调整字体小，放置*图例*在图的右上角
        '''
        font = font_manager.FontProperties(size='small')
        self._ax.legend(loc='upper right', prop=font)

    def _positions(self, count):
        '''
        对于给定的*count*位置数，获取位置数组。
        '''
        end = count * Gantt.POS_STEP + Gantt.POS_START
        pos = np.arange(Gantt.POS_START, end, Gantt.POS_STEP)
        return pos

    def show(self):
        self._plot_bars()
        self._configure_figure()
        plt.show()


if __name__ == '__main__':
    TEST_DATA = (
                 { 'label': "调研",'start':'2021-06-01 12:00:00', 'end': '2021-06-20 12:00:00'},  # @IgnorePep8
                 { 'label': "研发",'start':'2021-06-21 12:00:00', 'end': '2021-08-15 12:00:00'},  # @IgnorePep8
                 { 'label': "测试",'start':'2021-08-16 12:00:00', 'end': '2021-09-30 12:00:00'},  # @IgnorePep8
                 { 'label': "试用",'start':'2021-10-01 12:00:00', 'end': '2021-10-31 12:00:00'},  # @IgnorePep8
                )

    gantt = Gantt(TEST_DATA)
    gantt.show()


'''
data = [dict(Task = "调研", Start = '2021-06-01', End = '2021-06-20'),   
           dict(Task = "研发", Start = '2021-06-21', End = '2021-08-15'),   
           dict(Task = "测试", Start = '2021-08-16', End = '2021-09-30'),   
           dict(Task = "试用", Start = '2021-10-01', End = '2021-10-31')]

'''
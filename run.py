# coding:utf-8

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtawesome
from PyQt5.QtCore import pyqtSlot
from get_predict import pretreat,SentimentLstm,find_friend,SelfAttention
from aspect_based import rate_tag_recommand as aspect
from aspect_based_for_user import sb_function_added_by_zpd as zpd
import random
import os
current_path = os.path.dirname(__file__)

score,label,review = aspect()
class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.tag = 100
        self.tag_for_label = 0

    def random_for_10(self):
        list = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
        num = random.sample(list, 10)  # 输出10个固定长度的组合字符
        str1 = ''
        value = str1.join(num)  # 将取出的十个随机数进行重新合并
        print(value)
        return value


    @pyqtSlot()
    #按下搜索框
    def on_click(self):
        text = self.right_bar_widget_search_input.text()
        #print(text)
        self.get_score_text.clear()
        self.find_friends_text.clear()
        self.sb_tag, self.sb_rec_review = zpd(text)

        self.get_score_text.append('我们猜测您对这部电影的评分是（1为讨厌，5为喜欢，程度递增）\n'+'\n'+
                                   str(pretreat(text))+'\n'+'\n'
                                   '检索到您的关键词为：\n'+'\n'+
                                   self.sb_tag[0]+'\t'+self.sb_tag[1]+'\t'+self.sb_tag[2]+'\t'
                                   )
        if self.num_for_search + 5 <= len(self.sb_rec_review):
            self.find_friends_text.append(
                'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search] + '\n' + '\n'
                'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 1] + '\n' + '\n'
                'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 2] + '\n' + '\n'
                'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 3] + '\n' + '\n'
                'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 4] + '\n' + '\n'
                '或许您会对上面这些沙雕网友感兴趣：\n'
            )
        self.num_for_search += 5
        self.tag = 99

        #锁死正负面按钮的功能
        self.tag_for_label = 99


    #翻页用 不同button不同翻页
    def on_click1(self):
        #网友翻页
        if self.tag == 99:
            self.find_friends_text.clear()
            if self.num_for_search + 5 <= len(self.sb_rec_review):
                self.find_friends_text.append(
                    'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search] + '\n' + '\n'
                    'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 1] + '\n' + '\n'
                    'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 2] + '\n' + '\n'
                    'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 3] + '\n' + '\n'
                    'ID:' + str(self.random_for_10) + '\n' + self.sb_rec_review[self.num_for_search + 4] + '\n' + '\n'
                    '或许您会对上面这些沙雕网友感兴趣：\n'
                )
            self.num_for_search += 5

        if self.tag == 0:
            text = self.right_bar_widget_search_input.text()

            self.friend_list = find_friend(text)
            if self.num+5 <= len(self.friend_list):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+self.friend_list[self.num]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+self.friend_list[self.num+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+self.friend_list[self.num+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+self.friend_list[self.num+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+self.friend_list[self.num+4]+'\n'+'\n'
                                              '或许您会对上面这些沙雕网友感兴趣：\n'
                                              )
            self.num += 5
        #标签翻页
        if self.tag == 1:

            if self.num1+5 <= len(review[0][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+4]+'\n'+'\n'
                                                  )
            self.num1 += 5
            self.tag = 1
        if self.tag == 2:

            if self.num2+5 <= len(review[1][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num1+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num1+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num1+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num1+4]+'\n'+'\n'
                                                  )
            self.num2 += 5
            self.tag = 2
        if self.tag == 3:

            if self.num3+5 <= len(review[2][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+4]+'\n'+'\n'
                                                  )
            self.num3 += 5
            self.tag = 3
        if self.tag == 4:

            if self.num4+5 <= len(review[3][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+4]+'\n'+'\n'
                                                  )
            self.num4 += 5
            self.tag = 4
        if self.tag == 5:

            if self.num5+5 <= len(review[4][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+4]+'\n'+'\n'
                                                  )
            self.num5 += 5
            self.tag = 5
        if self.tag == 6:

            if self.num6+5 <= len(review[5][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+4]+'\n'+'\n'
                                                  )
            self.num6 += 5
            self.tag = 6
        if self.tag == 7:

            if self.num7+5 <= len(review[6][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+4]+'\n'+'\n'
                                                  )
            self.num7 += 5
            self.tag = 7
        if self.tag == 8:

            if self.num8+5 <= len(review[0][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+4]+'\n'+'\n'
                                                  )
            self.num8 += 5
            self.tag = 8
        if self.tag == 9:

            if self.num9+5 <= len(review[1][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+4]+'\n'+'\n'
                                                  )
            self.num9 += 5
            self.tag = 9
        if self.tag == 10:

            if self.num10+5 <= len(review[2][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+4]+'\n'+'\n'
                                                  )
            self.num10 += 5
            self.tag = 10
        if self.tag == 11:

            if self.num11+5 <= len(review[3][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+4]+'\n'+'\n'
                                                  )
            self.num11 += 5
            self.tag = 11
        if self.tag == 12:

            if self.num12+5 <= len(review[4][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                    'ID:' + str(self.random_for_10) + '\n' + review[4][1][self.num12] + '\n' + '\n'
                                                                                               'ID:' + str(
                        self.random_for_10) + '\n' + review[4][1][self.num12 + 1] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[4][1][self.num12 + 2] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[4][1][self.num12 + 3] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[4][1][self.num12 + 4] + '\n' + '\n'
                )
            self.num12 += 5
            self.tag = 12
        if self.tag == 13:

            if self.num13+5 <= len(review[5][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                    'ID:' + str(self.random_for_10) + '\n' + review[5][1][self.num13] + '\n' + '\n'
                                                                                               'ID:' + str(
                        self.random_for_10) + '\n' + review[5][1][self.num13 + 1] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[5][1][self.num13 + 2] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[5][1][self.num13 + 3] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[5][1][self.num13 + 4] + '\n' + '\n'
                )
            self.num13 += 5
            self.tag = 13
        if self.tag == 14:

            if self.num14+5 <= len(review[6][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                    'ID:' + str(self.random_for_10) + '\n' + review[6][1][self.num14] + '\n' + '\n'
                                                                                               'ID:' + str(
                        self.random_for_10) + '\n' + review[6][1][self.num14 + 1] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[6][1][self.num14 + 2] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[6][1][self.num14 + 3] + '\n' + '\n'
                                                                                           'ID:' + str(
                        self.random_for_10) + '\n' + review[6][1][self.num14 + 4] + '\n' + '\n'
                )
            self.num14 += 5
            self.tag = 14

    def on_click_aspect1(self):

        if self.tag_for_label == 1:
            if self.num1+5 <= len(review[0][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[0][0][self.num1+4]+'\n'+'\n'
                                                  )
            self.num1 += 5
            self.tag = 1
        if self.tag_for_label == 2:
            if self.num2+5 <= len(review[1][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num2+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num2+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num2+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[1][0][self.num2+4]+'\n'+'\n'
                                                  )
            self.num2 += 5
            self.tag = 2
        if self.tag_for_label == 3:
            if self.num3 + 5 <= len(review[2][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+1]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+2]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+3]+'\n'+'\n'
                                                  'ID:'+str(self.random_for_10)+'\n'+review[2][0][self.num3+4]+'\n'+'\n'
                                                  )
            self.num3 += 5
            self.tag = 3
        if self.tag_for_label == 4:
            if self.num4+5 <= len(review[3][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][0][self.num4+4]+'\n'+'\n'
                                              )
            self.num4 += 5
            self.tag = 4
        if self.tag_for_label == 5:
            if self.num5+5 <= len(review[4][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][0][self.num5+4]+'\n'+'\n'
                                              )
            self.num5 += 5
            self.tag = 5
        if self.tag_for_label == 6:
            if self.num6 + 5 <= len(review[5][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][0][self.num6+4]+'\n'+'\n'
                                              )
            self.num6 += 5
            self.tag = 6
        if self.tag_for_label == 7:
            if self.num7+5 <= len(review[6][0]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][0][self.num7+4]+'\n'+'\n'
                                              )
            self.num7 += 5
            self.tag = 7


    def on_click_aspect2(self):

        if self.tag_for_label == 1:
            if self.num8 + 5 <= len(review[0][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[0][1][self.num8+4]+'\n'+'\n'
                                              )
            self.num8 += 5
            self.tag = 8
        if self.tag_for_label == 2:
            if self.num9 + 5 <= len(review[1][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[1][1][self.num9+4]+'\n'+'\n'
                                              )
            self.num9 += 5
            self.tag = 9
        if self.tag_for_label == 3:
            if self.num10+5 <= len(review[2][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[2][1][self.num10+4]+'\n'+'\n'
                                              )
            self.num10 += 5
            self.tag = 10
        if self.tag_for_label == 4:
            if self.num11 + 5 <= len(review[3][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[3][1][self.num11+4]+'\n'+'\n'
                                              )
            self.num11 += 5
            self.tag = 11
        if self.tag_for_label == 5:
            if self.num12 + 5 <= len(review[4][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][1][self.num12]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][1][self.num12+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][1][self.num12+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][1][self.num12+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[4][1][self.num12+4]+'\n'+'\n'
                                              )
            self.num12 += 5
            self.tag = 12
        if self.tag_for_label == 6:
            if self.num13 + 5 <= len(review[5][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][1][self.num13]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][1][self.num13+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][1][self.num13+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][1][self.num13+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[5][1][self.num13+4]+'\n'+'\n'
                                              )
            self.num13 += 5
            self.tag = 13
        if self.tag_for_label == 7:
            if self.num14 + 5 <= len(review[6][1]):
                self.find_friends_text.clear()
                self.find_friends_text.append(
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][1][self.num14]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][1][self.num14+1]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][1][self.num14+2]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][1][self.num14+3]+'\n'+'\n'
                                              'ID:'+str(self.random_for_10)+'\n'+review[6][1][self.num14+4]+'\n'+'\n'
                                              )
            self.num14 += 5
            self.tag = 14



    #七个标签按钮
    def on_click_label1(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影主题标签为：')
        for i in range(len(label['主题'])):
            self.find_friends_text.append(label['主题'][i])
        self.find_friends_text.append('该电影主题综合得分：'+str(score['主题']))
        self.tag_for_label = 1

    def on_click_label2(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影剧情标签为：')
        for i in range(len(label['剧情'])):
            self.find_friends_text.append(label['剧情'][i])
        self.find_friends_text.append('该电影剧情综合得分：'+str(score['剧情']))
        self.tag_for_label = 2

    def on_click_label3(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影配乐标签为：')
        for i in range(len(label['配乐'])):
            self.find_friends_text.append(label['配乐'][i])
        self.find_friends_text.append('该电影配乐综合得分：'+str(score['配乐']))
        self.tag_for_label = 3

    def on_click_label4(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影画面标签为：')
        for i in range(len(label['画面'])):
            self.find_friends_text.append(label['画面'][i])
        self.find_friends_text.append('该电影画面综合得分：'+str(score['画面']))
        self.tag_for_label = 4

    def on_click_label5(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影节奏标签为：')
        for i in range(len(label['节奏'])):
            self.find_friends_text.append(label['节奏'][i])
        self.find_friends_text.append('该电影节奏综合得分：'+str(score['节奏']))
        self.tag_for_label = 5

    def on_click_label6(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影手法标签为：')
        for i in range(len(label['手法'])):
            self.find_friends_text.append(label['手法'][i])
        self.find_friends_text.append('该电影手法综合得分：'+str(score['手法']))
        self.tag_for_label = 6

    def on_click_label7(self):
        self.find_friends_text.clear()
        self.find_friends_text.append('经网友投票，该电影演技标签为：')
        for i in range(len(label['演技'])):
            self.find_friends_text.append(label['演技'][i])
        self.find_friends_text.append('该电影演技综合得分：'+str(score['演技']))
        self.tag_for_label = 7

    def init_ui(self):
        self.friend_list = []
        #num用来翻页用
        self.num = 0
        self.num1 = 0
        self.num2 = 0
        self.num3 = 0
        self.num4 = 0
        self.num5 = 0
        self.num6 = 0
        self.num7 = 0
        self.num8 = 0
        self.num9 = 0
        self.num10 = 0
        self.num11 = 0
        self.num12 = 0
        self.num13 = 0
        self.num14 = 0

        self.num_for_search = 0

        self.setFixedSize(960, 700)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_label_1 = QtWidgets.QPushButton("查看标签")
        self.left_label_1.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.left_label_3.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.home', color='white'), "主题")
        self.left_button_1.setObjectName('left_button')
        self.left_button_1.setToolTip("This is an example button")
        self.left_button_1.clicked.connect(self.on_click_label1)
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.paint-brush', color='white'), "剧情")
        self.left_button_2.setObjectName('left_button')
        self.left_button_2.setToolTip("This is an example button")
        self.left_button_2.clicked.connect(self.on_click_label2)
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "配乐")
        self.left_button_3.setObjectName('left_button')
        self.left_button_3.setToolTip("This is an example button")
        self.left_button_3.clicked.connect(self.on_click_label3)
        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.photo', color='white'), "画面")
        self.left_button_4.setObjectName('left_button')
        self.left_button_4.setToolTip("This is an example button")
        self.left_button_4.clicked.connect(self.on_click_label4)
        self.left_button_5 = QtWidgets.QPushButton(qtawesome.icon('fa.heartbeat', color='white'), "节奏")
        self.left_button_5.setObjectName('left_button')
        self.left_button_5.setToolTip("This is an example button")
        self.left_button_5.clicked.connect(self.on_click_label5)
        self.left_button_6 = QtWidgets.QPushButton(qtawesome.icon('fa.paw', color='white'), "手法")
        self.left_button_6.setObjectName('left_button')
        self.left_button_6.setToolTip("This is an example button")
        self.left_button_6.clicked.connect(self.on_click_label6)
        self.left_button_7 = QtWidgets.QPushButton(qtawesome.icon('fa.comment', color='white'), "演技")
        self.left_button_7.setObjectName('left_button')
        self.left_button_7.setToolTip("This is an example button")
        self.left_button_7.clicked.connect(self.on_click_label7)
        self.left_button_8 = QtWidgets.QPushButton(qtawesome.icon('fa.comment', color='white'), "反馈建议")
        self.left_button_8.setObjectName('left_button')
        self.left_button_9 = QtWidgets.QPushButton(qtawesome.icon('fa.question', color='white'), "遇到问题")
        self.left_button_9.setObjectName('left_button')

        self.left_xxx = QtWidgets.QPushButton(" ")
        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_6, 7, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_7, 8, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_3, 9, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_8, 10, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_9, 11, 0, 1, 3)
        self.right_bar_widget = QtWidgets.QWidget()  # 右侧顶部搜索框部件
        self.right_bar_layout = QtWidgets.QGridLayout()  # 右侧顶部搜索框网格布局
        self.right_bar_widget.setLayout(self.right_bar_layout)

        self.search_icon = QtWidgets.QPushButton(chr(0xf002) + ' ' + '搜索  ')
        self.search_icon.setToolTip("This is an example button")
        self.search_icon.clicked.connect(self.on_click)
        self.search_icon.setFont(qtawesome.font('fa', 16))
        self.right_bar_widget_search_input = QtWidgets.QLineEdit()
        self.right_bar_widget_search_input.setPlaceholderText("输入影评获取您的预测分，点击按钮进行搜索")

        self.right_bar_layout.addWidget(self.search_icon, 0, 0, 1, 1)
        self.right_bar_layout.addWidget(self.right_bar_widget_search_input, 0, 1, 1, 8)

        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 1, 9)

        score_from_aspect = "《海王》"
        self.right_recommend_label = QtWidgets.QLabel(score_from_aspect)
        self.right_recommend_label.setObjectName('right_lable')

        self.right_recommend_widget = QtWidgets.QWidget()  # 推荐封面部件
        self.right_recommend_layout = QtWidgets.QGridLayout()  # 推荐封面网格布局
        self.right_recommend_widget.setLayout(self.right_recommend_layout)

        self.recommend_button_1 = QtWidgets.QToolButton()
        self.recommend_button_1.setText("张沛东")  # 设置按钮文本
        self.recommend_button_1.setIcon(QtGui.QIcon(current_path + '\\r1.jpg'))  # 设置按钮图标
        self.recommend_button_1.setIconSize(QtCore.QSize(100, 100))  # 设置图标大小
        self.recommend_button_1.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # 设置按钮形式为上图下文

        self.recommend_button_2 = QtWidgets.QToolButton()
        self.recommend_button_2.setText("陈思哲")
        self.recommend_button_2.setIcon(QtGui.QIcon(current_path + '\\r2.jpg'))
        self.recommend_button_2.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_3 = QtWidgets.QToolButton()
        self.recommend_button_3.setText("吴昊")
        self.recommend_button_3.setIcon(QtGui.QIcon(current_path + '\\r3.jpg'))
        self.recommend_button_3.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_3.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_4 = QtWidgets.QToolButton()
        self.recommend_button_4.setText("高岳")
        self.recommend_button_4.setIcon(QtGui.QIcon(current_path + '\\r4.jpg'))
        self.recommend_button_4.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_4.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.recommend_button_5 = QtWidgets.QToolButton()
        self.recommend_button_5.setText("吴桦健")
        self.recommend_button_5.setIcon(QtGui.QIcon(current_path + '\\r5.jpg'))
        self.recommend_button_5.setIconSize(QtCore.QSize(100, 100))
        self.recommend_button_5.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.right_recommend_layout.addWidget(self.recommend_button_1, 0, 0)
        self.right_recommend_layout.addWidget(self.recommend_button_2, 0, 1)
        self.right_recommend_layout.addWidget(self.recommend_button_3, 0, 2)
        self.right_recommend_layout.addWidget(self.recommend_button_4, 0, 3)
        self.right_recommend_layout.addWidget(self.recommend_button_5, 0, 4)

        self.right_layout.addWidget(self.right_recommend_label, 1, 0, 1, 9)
        self.right_layout.addWidget(self.right_recommend_widget, 2, 0, 2, 9)

        self.right_newsong_lable = QtWidgets.QLabel("功能区")
        self.right_newsong_lable.setObjectName('right_lable')

        self.right_playlist_lable = QtWidgets.QLabel("大家在看")
        self.right_playlist_lable.setObjectName('right_lable')

        self.right_newsong_widget = QtWidgets.QWidget()  # 功能区部件
        self.right_newsong_layout = QtWidgets.QGridLayout()  # 功能区部件网格布局
        self.right_newsong_widget.setLayout(self.right_newsong_layout)
        self.right_newsong_widget1 = QtWidgets.QWidget()  # 功能区部件
        self.right_newsong_layout1 = QtWidgets.QGridLayout()  # 功能区部件网格布局
        self.right_newsong_widget1.setLayout(self.right_newsong_layout1)

        self.get_score_text = QtWidgets.QTextEdit()
        self.get_score_text.setFontWeight(75)
        self.get_score_text.setFontPointSize(10)
        self.newsong_button_1 = QtWidgets.QPushButton("点击这里以刷新沙雕网友列表")
        self.newsong_button_1.setToolTip("This is an example button")
        self.newsong_button_1.clicked.connect(self.on_click1)

        self.find_friends_text = QtWidgets.QTextEdit()
        self.find_friends_text.setFontWeight(75)
        self.find_friends_text.setFontPointSize(10)

        self.aspect1 = QtWidgets.QPushButton("该标签下正面评价")
        self.aspect1.resize(10,10)
        self.aspect1.setToolTip("")
        self.aspect1.clicked.connect(self.on_click_aspect1)
        self.aspect2 = QtWidgets.QPushButton("该标签下负面评价")
        self.aspect2.resize(10, 10)
        self.aspect2.setToolTip("")
        self.aspect2.clicked.connect(self.on_click_aspect2)
        #self.aspect3 = QtWidgets.QPushButton("画面好")
        #self.aspect3.resize(10, 10)
        #self.aspect3.setToolTip("")
        #self.aspect3.clicked.connect(self.on_click_aspect3)
        #self.aspect4 = QtWidgets.QPushButton("画面差")
        #self.aspect4.resize(10, 10)
        #self.aspect4.setToolTip("")
        #self.aspect4.clicked.connect(self.on_click_aspect4)
        #self.aspect5 = QtWidgets.QPushButton("手法好")
        #self.aspect5.resize(10, 10)
        #self.aspect5.setToolTip("")
        #self.aspect5.clicked.connect(self.on_click_aspect5)
        #self.aspect6 = QtWidgets.QPushButton("手法差")
        #self.aspect6.resize(10, 10)
        #self.aspect6.setToolTip("")
        #self.aspect6.clicked.connect(self.on_click_aspect6)

        self.right_newsong_layout.addWidget(self.get_score_text, 0, 1, )
        self.right_newsong_layout.addWidget(self.newsong_button_1, 1, 1, )
        self.right_newsong_layout.addWidget(self.find_friends_text, 2, 1, )
        self.right_newsong_layout1.addWidget(self.aspect1, 3, 0, )
        self.right_newsong_layout1.addWidget(self.aspect2, 3, 1, )
        #self.right_newsong_layout1.addWidget(self.aspect3, 3, 2, )
        #self.right_newsong_layout1.addWidget(self.aspect4, 3, 3, )
        #self.right_newsong_layout1.addWidget(self.aspect5, 3, 4, )
        #self.right_newsong_layout1.addWidget(self.aspect6, 3, 5, )


        self.right_playlist_widget = QtWidgets.QWidget()  # 播放歌单部件
        self.right_playlist_layout = QtWidgets.QGridLayout()  # 播放歌单网格布局
        self.right_playlist_widget.setLayout(self.right_playlist_layout)

        self.playlist_button_1 = QtWidgets.QToolButton()
        self.playlist_button_1.setText("蜘蛛侠")
        self.playlist_button_1.setIcon(QtGui.QIcon(current_path + '\\p1.jpg'))
        self.playlist_button_1.setIconSize(QtCore.QSize(85, 85))
        self.playlist_button_1.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_2 = QtWidgets.QToolButton()
        self.playlist_button_2.setText("海王")
        self.playlist_button_2.setIcon(QtGui.QIcon(current_path + '\\p2.jpg'))
        self.playlist_button_2.setIconSize(QtCore.QSize(85, 85))
        self.playlist_button_2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_3 = QtWidgets.QToolButton()
        self.playlist_button_3.setText("狗十三")
        self.playlist_button_3.setIcon(QtGui.QIcon(current_path + '\\p3.jpg'))
        self.playlist_button_3.setIconSize(QtCore.QSize(85, 85))
        self.playlist_button_3.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.playlist_button_4 = QtWidgets.QToolButton()
        self.playlist_button_4.setText("无名之辈")
        self.playlist_button_4.setIcon(QtGui.QIcon(current_path + '\\p4.jpg'))
        self.playlist_button_4.setIconSize(QtCore.QSize(85, 85))
        self.playlist_button_4.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

        self.right_playlist_layout.addWidget(self.playlist_button_1, 0, 0)
        self.right_playlist_layout.addWidget(self.playlist_button_2, 0, 1)
        self.right_playlist_layout.addWidget(self.playlist_button_3, 1, 0)
        self.right_playlist_layout.addWidget(self.playlist_button_4, 1, 1)

        self.right_layout.addWidget(self.right_newsong_lable, 4, 0, 1, 5)
        self.right_layout.addWidget(self.right_playlist_lable, 4, 5, 1, 4)
        self.right_layout.addWidget(self.right_newsong_widget, 5, 0, 1, 5)
        self.right_layout.addWidget(self.right_playlist_widget, 5, 5, 1, 4)
        self.right_layout.addWidget(self.right_newsong_widget1, 8, 0, 1, 9)

        self.right_process_bar = QtWidgets.QProgressBar()  # 播放进度部件
        self.right_process_bar.setValue(49)
        self.right_process_bar.setFixedHeight(3)  # 设置进度条高度
        self.right_process_bar.setTextVisible(False)  # 不显示进度条文字

        self.right_playconsole_widget = QtWidgets.QWidget()  # 播放控制部件
        self.right_playconsole_layout = QtWidgets.QGridLayout()  # 播放控制部件网格布局层
        self.right_playconsole_widget.setLayout(self.right_playconsole_layout)

        self.console_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.backward', color='#F76677'), "")
        self.console_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.forward', color='#F76677'), "")
        self.console_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.pause', color='#F76677', font=18), "")
        self.console_button_3.setIconSize(QtCore.QSize(30, 30))

        self.right_playconsole_layout.addWidget(self.console_button_1, 0, 0)
        self.right_playconsole_layout.addWidget(self.console_button_2, 0, 2)
        self.right_playconsole_layout.addWidget(self.console_button_3, 0, 1)
        self.right_playconsole_layout.setAlignment(QtCore.Qt.AlignCenter)  # 设置布局内部件居中显示

        self.right_layout.addWidget(self.right_process_bar, 9, 0, 1, 9)
        self.right_layout.addWidget(self.right_playconsole_widget, 10, 0, 1, 9)

        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        ''')

        self.right_bar_widget_search_input.setStyleSheet(
            '''
            QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
            }''')
        self.right_bar_widget_search_input.setStyleSheet(
            '''
            QPushButton{border:none;color:white;}

            }''')

        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        self.right_recommend_widget.setStyleSheet(
            '''
                QToolButton{border:none;}
                QToolButton:hover{border-bottom:2px solid #F76677;}
            ''')
        self.right_playlist_widget.setStyleSheet(
            '''
                QToolButton{border:none;}
                QToolButton:hover{border-bottom:2px solid #F76677;}
            ''')

        self.right_newsong_widget.setStyleSheet('''
            QPushButton{
                border:none;
                color:gray;
                font-size:12px;
                height:40px;
                padding-left:5px;
                padding-right:10px;
                text-align:left;
            }
            QPushButton:hover{
                color:black;
                border:1px solid #F3F3F5;
                border-radius:10px;
                background:LightGray;
            }
        ''')

        self.right_newsong_widget1.setStyleSheet('''
            QPushButton{
                border:none;
                color:gray;
                font-size:12px;
                height:40px;
                padding-left:5px;
                padding-right:10px;
                text-align:left;
            }
            QPushButton:hover{
                color:black;
                border:1px solid #F3F3F5;
                border-radius:10px;
                background:LightGray;
            }
        ''')

        self.right_process_bar.setStyleSheet('''
            QProgressBar::chunk {
                background-color: #F76677;
            }
        ''')

        self.right_playconsole_widget.setStyleSheet('''
            QPushButton{
                border:none;
            }
        ''')

        self.setWindowOpacity(0.9)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框

        self.main_layout.setSpacing(0)




def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.newsong_button_1.setCheckable(True)
    gui.aspect1.setCheckable(True)
    gui.aspect2.setCheckable(True)
    #gui.aspect3.setCheckable(True)
    #gui.aspect4.setCheckable(True)
    #gui.aspect5.setCheckable(True)
    #gui.aspect6.setCheckable(True)

    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
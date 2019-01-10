
import scipy.io as sio
import numpy as np
import csv
import os
import pickle
import numpy as np
import pandas as pd
import sys
import importlib

importlib.reload(sys)


current_path = os.path.dirname(__file__) + "\\data"
if __name__=='__main__':
    #current_path = "F:\desktop\web\movie_initial_reviews"
    task = []
    # reviews = [sentence1,sentence2,...]
    # sentence = [word1,word2,...] (without word out of Chinese)

    # scores = {'score': [review sentences in this score]}

    for root, dirs, files in os.walk(current_path):
        for file in files:
            # read the csv files in different encoding
            if os.path.splitext(file)[1] != '.csv': continue


            csvFile = open(root + '\\' + file, "r")
            reader = csv.reader(csvFile)
            for item in reader:
                task.append(item[0])


            print("\nSuccessfully collecting data for",os.path.splitext(file)[0])
            # print("Current reviews:",len(reviews),"\n")


    print(task[5])


    print(len(task))

    np.save('task_reviews.npy', task)



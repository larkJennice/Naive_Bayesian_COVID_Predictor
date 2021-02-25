from collections import defaultdict
from datetime import date
import pandas as pd
import numpy as np

import sys 
import math
import time

class naive_bayes_classifier:
    def __init__(self, num_class, size):
        self.num_class = num_class
#         self.likelihood = defaultdict(lambda: np.zeros(num_class))
# #         self.prior = np.zeros(num_class)
# #         self.conditioncount = defaultdict(lambda:np.zeros(self.num_class))
        self.countsurvive = defaultdict(lambda:np.zeros(self.num_class))
        self.countdied = defaultdict(lambda:np.zeros(self.num_class))
        self.surviveratio = 0
        self.deadratio = 0
        self.sur = 0
        self.de = 0
        self.size = size
            
    def I(self,a,b):
        if a == 0 or b == 0:
            return 0
        sum_num = a+b
        result = -(a/sum_num)*math.log(a/sum_num,2)-(b/sum_num)*math.log(b/sum_num,2)
        return result   

    def calculate_entropy(self, countsurvive, countdied):
        survive_num = countsurvive[0][2]
        died_num = countdied[1][2]
        Y = self.I(survive_num,died_num)
        print("Y",Y)
        for i in range(self.num_class):
            entropy = 0
            for j in range(len(countsurvive)):
                sum_en = countsurvive[j][i] + countdied[j][i]
                entropy += (sum_en/self.size)* self.I(countsurvive[j][i],countdied[j][i])
            entropy = Y - entropy
            print(i,entropy)
#         return ((a/self.size)*I(b,c))
        
    def train(self,data):
        conditioncount = defaultdict(lambda:np.zeros(self.num_class))
#         eachclass = np.zeros(self.num_class)
        labelcount = np.zeros(self.num_class)
        countsurvive = defaultdict(lambda:np.zeros(self.num_class))
        countdied = defaultdict(lambda:np.zeros(self.num_class))
#         collectCondition = set()
        survivediedratio = [0,0]
        #get entropy for death
            
        for(conditions, label) in data:
            for con in range(len(conditions)): 
#                 conditioncount[conditions[con]][label] += 1
                if(data[2][0][con]==0):
                    countsurvive[conditions[con]][label] += 1
                elif(data[2][0][con]==1):
                    countdied[conditions[con]][label] += 1 
#                 collectCondition.add(conditions[con]) 

#         self.calculate_entropy(countsurvive,countdied)
    
        for i in range(len(data[2][0])):
                if data[2][0][i] == 1:
                    survivediedratio[1] += 1
                elif data[2][0][i] == 0:
                    survivediedratio[0] += 1
        self.sur = survivediedratio[0]
        self.de = survivediedratio[1]
        for i in range(len(countsurvive)):
            countsurvive[i] = countsurvive[i]/survivediedratio[0]
        for i in range(len(countdied)):
            countdied[i] = countdied[i]/survivediedratio[1]
    
        self.countsurvive = countsurvive
        self.countdied = countdied
        
        self.surviveratio = survivediedratio[0]/(survivediedratio[0]+survivediedratio[1])
        self.deadratio = survivediedratio[1]/(survivediedratio[0]+survivediedratio[1])

    def predict(self,data):
        prediction = list()
        for i in range(len(data[0][0])):
            resSurvive = self.surviveratio
            resDied = self.deadratio
            for j in range(len(data)):
                if j == 2: #skip the 'date_died' column 
                    continue
                if (self.countsurvive[data[j][0][i]][j] != 0):
                    resSurvive *= self.countsurvive[data[j][0][i]][j]
                else:
                    resSurvive *= 1/(self.sur+1*self.size)
                if (self.countsurvive[data[j][0][i]][j] != 0):
                    resDied *= self.countdied[data[j][0][i]][j]
                else:
                    resSurvive *= 1/(self.de+1*self.size)
            if (resSurvive > resDied):
                prediction.append(0)
            else:
                prediction.append(1)
        
        count_acc = 0
        for i in range(len(data[2][0])):
            if (prediction[i] == data[2][0][i]):
                count_acc += 1
        return prediction, (count_acc/len(data[2][0]))
        
        
         
#data preprocessing
def calculate_entry_symptoms(entry,symptom):
    days = list()
    for i in range(len(entry)):
        day_e, month_e, year_e = entry[i].split('-')
        day_s, month_s, year_s = symptom[i].split('-')
        f_date = date(int(year_s), int(month_s), int(day_s))
        l_date = date(int(year_e), int(month_e), int(day_e))
        delta = l_date - f_date
        days.append(delta)
    return days

def loadfile(file):
    df = pd.read_csv(file)
    df['entry_symptoms_days'] = calculate_entry_symptoms(df['entry_date'],df['date_symptoms'])
    df.drop(['entry_date','date_symptoms'],axis=1,inplace=True)
    df_nice = df.columns.tolist()
    load = list()
    day = list()
    len_column = 0
    for column in df.columns.tolist():
        df_value = df[column].tolist()
        if column == 'date_died':
            for i in range(len(df_value)):
                if df_value[i] == '9999-99-99':
                    df_value[i] = 0
                else:
                    df_value[i] = 1
#             print(df_value)
        load.append([df_value,len_column])
        len_column+=1
    return load

if __name__ == '__main__':
    classifier = naive_bayes_classifier(21,283301)
# #     start = time.time()
#     classifier.train(loadfile('covid_train.csv'))
# #     end = time.time()
# #     print(end - start)
# #     start = time.time()
#     res, acc = classifier.predict(loadfile('covid_valid.csv'))
# #     end = time.time()
# #     print(end - start)
    classifier.train(loadfile(sys.argv[1]))
    res, acc = classifier.predict(loadfile(sys.argv[2]))

    for i in res: 
        print(str(i))
        
#     print("accuracy", acc)
#     var = sys.stdout 
#     for i in labels: 
#         var.write(i) 

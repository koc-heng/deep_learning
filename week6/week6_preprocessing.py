import pandas as pd 
import numpy as np

# etl data / data mining is not in the scope of this code
data = pd.read_csv('titanic.csv')
#data.info()
#data.describe()
#data.isnull().sum()

# 重新計算一下價格
data['new_price'] = data.groupby('Ticket')['Fare'].transform('mean')

# 替換一下稱謂
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

data['Title']=data['Title'].replace({'Ms':'Miss','Mlle':'Miss','Mme':'Mrs'})
data['Title']=data['Title'].replace(['Sir','Don','Dona','Jonkheer','Lady','Countess'], 'Noble')
data['Title']=data['Title'].replace(['Dr', 'Rev','Col','Major','Capt'], 'Others')
data['Title'].value_counts()

#從稱謂中get年齡的中位數，並且去補值
title_age_median = data.groupby('Title')['Age'].median()

def fill_missing_age(row):
    if pd.isnull(row['Age']):
        return title_age_median[row['Title']]
    else:
        return row['Age']

data['Age'] = data.apply(fill_missing_age, axis=1)

#算一下新的var 家庭大小
data['F_Size'] = data['SibSp'] + data['Parch'] + 1
data[['F_Size', 'Survived']].groupby(['F_Size'], as_index=False).mean()

#分成2類
def family_size_category(family_size):
    if  2<= family_size <= 4:
        return 'H_S_R'
    else:
        return 'L_S_R'

data['F_Size'] = data['F_Size'].map(family_size_category)

#把數值型轉成類別型
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

data["Title"] = data["Title"].astype('category').cat.codes
data["F_Size"] = data["F_Size"].astype('category').cat.codes

#最後取出我想要的相關資料，並輸出csv
data_pre = data[['Survived','Pclass','Sex','Age', 'SibSp', 'Parch', 'new_price', 'Title', 'F_Size']]
data_pre.to_csv('titanic_preprocessed.csv', index=False)
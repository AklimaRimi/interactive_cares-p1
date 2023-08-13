import joblib
import os
from PyPDF2 import PdfReader
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filePath', type=str, help="Enter path")
args = parser.parse_args()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

words = stopwords.words('english')

vector = CountVectorizer()

lem = WordNetLemmatizer()
stem = PorterStemmer()
def pre(s):
  s = s.lower()
  s = re.sub(r'\d+',' ',s)
  s = re.sub(r'[^\w\s]',' ',s)
  s = re.sub(r'_',' ',s)
  s = s.split()
  st = [lem.lemmatize(val) for val in s]
  st = [stem.stem(val) for val in st]
  return st

def filter(li):
  filter_list = []
  for x in li:
    if x not in words:
      x = pre(x)
      if len(x) > 0:
         filter_list.append(x)
  return filter_list

cols = []
with open('col_name.txt','r') as f:
    txt = f.read().split('\n')
    
    for i in txt:
        cols.append(i)
model = joblib.load('rf3.joblib')


    
li = ['INFORMATION-TECHNOLOGY','BUSINESS-DEVELOPMENT','FINANCE','ADVOCATE','ACCOUNTANT','ENGINEERING','CHEF','AVIATION',
        'FITNESS','SALES','BANKING','HEALTHCARE','CONSULTANT','CONSTRUCTION','PUBLIC-RELATIONS','HR','DESIGNER',
        'ARTS','TEACHER','APPAREL','DIGITAL-MEDIA','AGRICULTURE','AUTOMOBILE','BPO']


categorized_resume = []
filename = args.filePath
def categorize():
    for l in os.listdir(filename):

        pdf_path = os.path.join(filename,l)
        reader = PdfReader(pdf_path)
        pdf = reader.pages[0].extract_text()

    

        df = pd.DataFrame()
        df['Tokens'] = [pdf]

        df['Tokens'] = df['Tokens'].apply(lambda x: word_tokenize(x))

        df['Tokens'] = df['Tokens'].apply(lambda x: filter(x))
        df['Tokens'] = df['Tokens'].apply(lambda skills: ' '.join([' '.join(skill) for skill in skills]))

        word_count  = vector.fit_transform(df['Tokens'])
        word_count_df = pd.DataFrame(word_count.toarray(), columns=vector.get_feature_names_out())

        filtered_cols = []

        for i in word_count_df.columns:
            if i in cols:
                filtered_cols.append(i)
        word_count_df = word_count_df[filtered_cols]
        df = pd.DataFrame()
        def add_col():
            col = word_count_df.columns
            for i in cols:
                if i not in col:
                    df[i] = [0]
                else:
                    df[i] = word_count_df[i]                
                
        add_col()

                
        pred = model.predict(df)
        name = li[pred[0]]
        print(name)
        
        os.makedirs(filename+'/'+name,exist_ok=True)
        os.replace(pdf_path,filename+'/'+name+'/'+l)
        categorized_resume.append([l,name])
    df = pd.DataFrame(categorized_resume,columns=['filename','category'])
    df.to_csv('categorized_resume.csv',index = False)
    
    
if __name__ == '__main__':
    categorize()

# coding: utf-8

# In[1]:


import glob
import io
import os
import pdb
import sys

import re
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.metrics.pairwise import cosine_similarity
#import xgboost


# In[2]:


def train_tfidf(filepath):
    documents = []
    for thefile in glob.glob(filepath)[:5000]:
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            documents.append(text)
    
    tfidf_vectorizer = TfidfVectorizer()
    x = tfidf_vectorizer.fit_transform(documents)
    idf = tfidf_vectorizer.idf_
    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), idf))
    return tfidf_dict
#f = sorted(m.items(), key = lambda kv :kv[1], reverse = True)

tfidf_dict = train_tfidf("text-test/*.txt")


# In[3]:


block = '\u2588'
outpu_path = 'text-test/redacted'
def write_redacted(text, names, thefile):
    
    data = text
    f_name = thefile.replace('text-test', 'text-test/redacted')    
    for elm in names:
        m = len(elm)
        elm = r'\b' + elm + r'\b'
        bl = m * block
        data = re.sub(elm, bl, data)
    with open(f_name, 'w', encoding = "utf-8")as file:
            file.write(data)
            file.close()


# In[4]:


def get_entity(text):
    names = []
    redact_names = []
    """Prints the entity inside of the text."""
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                redact_names.extend([leave[0] for leave in chunk.leaves()])
                names.append(' '.join(c[0] for c in chunk.leaves()))
    return names, redact_names


# In[5]:


def doextraction(glob_text):
    names = []
    train_data = []
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text)[:200]:
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            train_data.extend(get_training_features(text, thefile))
            n , m = get_entity(text)
            m.sort(reverse=True, key=len)
            write_redacted(text,m,thefile)
            names.extend(n)
            
    return names, train_data


# In[6]:


def get_training_features(text,file):
    train_list = []
    #print(file)
    t = text + '!'
    #text = re.sub("(\\d|\\W)+", " ", t)
    #print(text) 
    # Write code here to extract important words if time permits
    doc_len = len(t)
    user_rating = int(re.findall(r'(\d{1,2}).txt', file)[0])
    for sent in sent_tokenize(t):
        m = 0
        chunks =  ne_chunk(pos_tag(word_tokenize(sent)))

        for chunk in chunks: 
            names_dict = {}
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name = ' '.join(c[0] for c in chunk.leaves())
                #n_l.append(name)
                if m == 0:
                    left = ''
                elif type(chunks[m-1][0]) is tuple:
                    left = chunks[m-1][0][0]
                else:
                    left = chunks[m-1][0]
                if type(chunks[m+1][0]) is tuple:
                    right = chunks[m+1][0][0]
                else:
                    right = chunks[m+1][0]
                len_name = len(name)
                
                if left in tfidf_dict.keys():
                    tfidf_left = tfidf_dict[left]
                else:
                    tfidf_left = 0

                if right in tfidf_dict.keys():
                    tfidf_right = tfidf_dict[right]
                else:
                    tfidf_right = 0
                no_spaces = name.count(' ')

                names_dict['name'] = name
                names_dict['name_length'] = len_name
                names_dict['spaces'] = no_spaces
                names_dict['left_name'] = tfidf_left
                names_dict['right_name'] = tfidf_right
                names_dict['len_chars'] = doc_len
                names_dict['rating'] = user_rating
                train_list.append(names_dict)
            m += 1
    
    no_names = len(train_list)
#     for elm in train_list: # Try to shift above
#         elm['no_names'] = no_names

    return train_list


# In[7]:


#text = "████████ ████ is a cartoon comedy. It ran at the same time as some other programs about school life, such as Teachers. My 35 years in the teaching profession lead me to believe that ████████ ████'s satire is much closer to reality than is Teachers. The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... ████. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to ████████ ████. I expect that many adults of my age think that ████████ ████ is far fetched. What a pity that it isn't!"
def get_testing_features(text, file):
    test_list = []
#    print(file)
    l = len(text)
    text = text + '!'
    user_rating = int(re.findall(r'(\d{1,2}).txt', file)[0])
    check_left = r'(\w*|\W)\s*'+ block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*'+ block + r'+'
    check = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+'
    check_right = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block +r'+\s*(\W{0,1}\w*)'
#    print(re.findall(check, text))
#    print(re.findall(check_left, text))
#    print(check_right)
#    print(re.findall(check_right, text))
#     for m in re.finditer(check, text):
#         print(m)
    name_blocks = re.findall(check, text)
    left = re.findall(check_left, text)
    right = re.findall(check_right, text)
    for i in range(len(name_blocks)):
        block_dict = {}
        if left[i] in tfidf_dict.keys():
            tfidf_left = tfidf_dict[left[i]]
        else:
            tfidf_left = 0

        if right[i] in tfidf_dict.keys():
            tfidf_right = tfidf_dict[right[i]]
        else:
            tfidf_right = 0
        block_dict['name_length'] = len(name_blocks[i])
        block_dict['spaces'] = name_blocks[i].count(' ')
        block_dict['left_name'] = tfidf_left
        block_dict['right_name'] = tfidf_right
        block_dict['len_chars'] = l
        block_dict['rating'] = user_rating
#        block_dict['no_names'] = len(name_blocks)
        #print(block_dict) 
        test_list.append(block_dict)
    
    return test_list, name_blocks


# In[8]:


def write_unredactor(test_data, text, name_blocks, filepath):
    dict_vectorizer = DictVectorizer()
    test_features = dict_vectorizer.fit_transform(test_data).toarray()
    probs = model.predict_proba(test_features)
    filepath = filepath.replace('redacted','unredacted')
    
    for i in range(len(probs)):
        top_5_idx = np.argsort(probs[i])[-5:]
        top_5_values = [f[i] for i in top_5_idx]
        
        text = text + "\n{}. For {} similar names are {}\n".format(i, name_blocks[i], [val for val in top_5_values])
        
    with open(filepath, 'w',encoding = "utf-8") as fy:
        fy.write(text)
        fy.close


# In[9]:


names, train_data = doextraction("text-test/*.txt")


# In[10]:


random.seed(5293)
random.shuffle(train_data)
tr_data = train_data
#tr_data


# In[11]:


train_target = []
for elm in tr_data:
    train_target.append(elm['name'])
    del elm['name']
#print(len(tr_data), len(train_target))
#tr_data


# In[13]:


dict_vectorizer = DictVectorizer()
train_features = dict_vectorizer.fit_transform(tr_data).toarray()

train_target = np.asarray(train_target)
model = ensemble.RandomForestClassifier()
model.fit(train_features, train_target)
#test_predict = model.predict(test_features)


# In[15]:


f = list(set(train_target))
#f = np.asarray(f)


# In[17]:


test_data = []
for thefile in glob.glob("text-test/redacted/*.txt")[:40]:
    with open(thefile, 'r', encoding='utf-8') as fyl:
        text = fyl.read()
        test_data, name_blocks =get_testing_features(text, thefile)
        if len(test_data) > 0:
            write_unredactor(test_data, text, name_blocks, thefile)


# # In[12]:
#
#
# random.shuffle(test_data)
# test_data
#
#
# # In[14]:
#
#
# print(list(set(test_predict)))
#
#
# # In[15]:
#
#
# for i in range(len(test_predict)):
#     print(test_data[i], "\nPredicted Name is {}".format(test_predict[i]))
#
#
# # In[16]:
#
#
# model =svm.SVC(probability = True)
# model.fit(train_features, train_target)
# test_predict = model.predict(test_features)
#
#
# # In[17]:
#
#
# print(len(set(train_target)))
#
#
# # In[18]:
#
#
# probs = model.predict_proba(test_features)
# probs[1]
#
#
# # In[26]:
#
#
# for i in range(len(probs)):
#     ind = np.argpartition(probs[i], -7)[-7:]
#     print(f[ind])
#
#
# # In[33]:
#
#
# top_2_idx = np.argsort(probs[5])[-3:]
# top_2_values = [f[i] for i in top_2_idx]
#
#
# # In[34]:
#
#
# top_2_values
#
#
# # In[ ]:
#
#
# ind = np.argpartition(probs[2], -7)[-7:]
# train_target[ind]
#
#
# # In[ ]:
#
#
# print(list(set(train_target))[481])
#
#
# # In[ ]:
#
#
# for i in range(len(test_predict)):
#     print(test_data[i], "\nPredicted Name is {}".format(test_predict[i]))
#
#
# # In[76]:
#
#
# with open('text-test/redacted/100_7.txt', 'r', encoding='utf-8') as fyl:
#     text = fyl.read()
#
#     m, nm = get_testing_features(text,'text-test/redacted/0_9.txt' )
#     write_unredactor(m, text, nm, 'text-test/redacted/0_9.txt')
#     #print(n)
#
#
# # In[78]:
#
#
# model = ensemble.RandomForestClassifier()
# model.fit(train_features, train_target)
# test_predict = model.predict_proba(test_features)
#
#
# # In[81]:
#
#
# len(test_predict[0])
#
#
# # In[ ]:
#
#
# print(list(set(test_predict)))
#
#
# # In[ ]:
#
#
# for i in range(len(test_predict)):
#     print(test_data[i], "\nPredicted Name is {}".format(test_predict[i]))
#
#
# # In[ ]:
#
#
# import unittest
#
# class TestNotebook(unittest.TestCase):
#
#     def test_entity(self):
#         self.assertEqual(cd /add0(2, 2), 5)
#
#
# # In[ ]:
#
#
# model = xgboost.XGBClassifier()
# model.fit(train_features, train_target)
# test_predict = model.predict(test_features)
#
#
# # In[ ]:
#
#
# print(list(set(test_predict)))
#
#
# # In[30]:
#
#
# for i in range(len(test_predict)):
#     print(test_data[i], "\nPredicted Name is {}".format(test_predict[i]))
#
#
# # In[54]:
#
#
# def get_feature_redacted(text):
#     feature=[]
#     w=0
#     w=len(nltk.word_tokenize(text))
#     count=0
#     space=0
#     count_word=0
#     count_Redacted=0
#     l=[]
#     for i in range(0,len(text)):
#         if text[i]=="*":
#             count_word=1
#             count+=1
#             #if text[i+1]==" " and text[i+2]=="*":
#                 #i=i+1
#                 #space+=1
#         else:
#             if text[i-1]=="*" and text[i+1]=="*":
#                 #i=i+1
#                 space+=1
#                 continue
#             if count>0:
#                 if space>=0:
#                     l.append(space)
#                 l.append(count)
#                 l.append(w)
#                 feature.append(l)
#                 l=[]
#             count=0
#             space=0
#             if count_word==1:
#                 count_Redacted+=1
#                 count_word=0
#         #print(i)
#     #print(count_Redacted)
#     #print(feature)
#     feature_actual=[]
#     for i in feature:
#         l=[]
#         l.append(count_Redacted)
#         for j in i:
#             l.append(j)
#         feature_actual.append(l)
#     #print(feature_actual)
#     return feature_actual
#
#
# # In[61]:
#
#
# with open('text-test/redacted/0_9.txt', 'r', encoding='utf-8') as fyl:
#     text = fyl.read()
#     m = get_feature_redacted(text)
#
#
# # In[62]:
#
#
# m
#
#
# # In[39]:
#
#
# with open('text-test/redacted/100_7.txt', 'r', encoding='utf-8') as fyl:
#     text = fyl.read()
#     m = get_testing_features(text, 'text-test/redacted/0_9.txt')
#     dict_vectorizer1 = DictVectorizer()
#
#     m_features = dict_vectorizer.transform(m).toarray()
#     m_target = np.asarray(m)
#     model1 = ensemble.RandomForestClassifier()
#     model1.fit(train_features, train_target)
#     m_predict = model1.predict(m_features)
#
#
# # In[40]:
#
#
# for i in range(len(m_predict)):
#     print(m[i], "\nPredicted Name is {}".format(m_predict[i]))
#
#
# # In[31]:
#
#
# val = [1, 2,3,4,5,6,7,100, 4, 120]
# ind = np.argpartition(val, -5)[-5:]
# ind
#
#
# # In[69]:
#
#
# len(train_features)
#

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

df = pd.read_excel (r'ngrams_frequencies_withNames.xlsx')
#df.dtypes
print(df[:11])
# -

# What do we get from the network? Top x characters? Multiply by character probability
# TODO:
# Get character frequency
# Calculate character probability: freq/tot
# Calculate word probability: freq/tot
# Calculate levenshtein distance:
# 1. create empty dict to save distances
# 2. check if word is in given dictionary
# 3. Yes? distance = 0, No? Find minimal distance
# 4. Find minimal distance:
# 5. delete / substitute / insert
#
#
#
# Steps:
# 1. check if word exists
# 2. if not, calculate edit distances
# 3. calculate final probability by min(edit_distance)*word_probability
# 4. return max(final_probability)

# +
df = pd.read_excel (r'ngrams_frequencies_withNames.xlsx')
print(df[:11])


# Create a dictionary with all characters
#df = df['Hebrew_character'].values.tolist()
df = df['Names'].values.tolist()
df = [word.split('_') for word in df]
#print(df)
character_dic = {}
for word in df:
    for letter in word:
        if letter not in character_dic:
            character_dic[letter] = 0
        character_dic[letter] += 1

        
print(character_dic.keys())        
for key, value in character_dic.items():
    print(str(key) + ' ' + str(value))    
#print(len(character_dic))


# +
df2 = df['Names'].values.tolist()
#df3 = [word.replace('_',' ') for word in df2]
#print (df3)

df4 = [word.split() for word in df3]
#print(df4)
#df5 = [word.append() for word in df4]
df5 = []
for words in df4:
    for word in words:
        df5.append(word)
dic1 = {}
for word in df5:
    if word not in dic1:
        dic1[word] = 0
    dic1[word] += 1
print(dic1)
print (len(dic1))
# -

for dic in sorted(dic1): print(dic + " " + str(dic1[dic]))


# +
def levenshtein(s1, s2):
    d = {}
    lens1 = len(s1)
    lens2 = len(s2)
    
    for i in range(-1,lens1+1):
        d[(i,-1)] = i+1
        
    for j in range(-1,lens2+1):
        d[(-1,j)] = j+1
 
    for i in range(lens1):
        for j in range(lens2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1,
                           d[(i,j-1)] + 1, 
                           d[(i-1,j-1)] + cost,
                          )
    return d[lens1-1,lens2-1]

ngrams = pd.read_excel (r'ngrams_frequencies_withNames.xlsx')
#df = ngrams['Hebrew_character'].values.tolist(df = ngrams['Hebrew_character'].values.tolist(')
#print(df[:11])


#TODO
# dont rely only on edit distance, take prob with mindist
# Think of proper formula

import math
def bestGuess(checkword):
    distlist = []
    mindist = float('inf')
    freqsum = ngrams['Frequencies'].sum()
    
    for row in ngrams.itertuples():
        word = row.Hebrew_character
        prob = round(row.Frequencies / freqsum * 100,6)
        dist = levenshtein(checkword, word)
        # test formula:
        dist = math.exp(dist)/prob
        #print (dist)
        if dist == mindist:
            #distlist.append([word,dist,prob])
            distlist.append([word,dist])
            
        if dist < mindist:
            #distlist = [[word,dist,prob]]
            distlist = [[word,dist]]
            mindist = dist
            
    return distlist


bestGuess('הימהימים')

# -



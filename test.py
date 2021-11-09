import pandas as pd
import string
import re
main_dict={}
def generate_ngrams(text,n):

    # split sentences into tokens
    tokens=re.split("\\s+",text)
    ngrams=[]

    # collect the n-grams
    for i in range(len(tokens)-n+1):
       temp=[tokens[j] for j in range(i,i+n)]
       ngrams.append(" ".join(temp))

    return ngrams

def generate_dict(ngrams):
    for i in ngrams:
        if(i in main_dict):
            main_dict[i]=main_dict[i]+1
        else:
            main_dict[i]=1

def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans

df = pd.read_csv('data.csv')
df['news_']= df['Text'].apply(lambda x:remove_punctuation(x))
df['news_ng']= df['Text'].apply(lambda x:generate_ngrams(x,2))




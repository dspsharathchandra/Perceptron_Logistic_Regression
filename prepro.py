import pandas as pd
import string
import re
main_dict={}
sw=["a", "about", "above", "after", "again", "against", "all", 
"am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", 
"been", "before", "being", "below", "between", "both", "but", 
"by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", 
"does", "doesnt", "doing", "dont", "down", "during", "each", 
"few", "for", "from", "further", "had", "hadnt", "has", "hasnt", 
"have", "havent", "having", "he", "hed", "hell", "hes", "her", 
"here", "heres", "hers", "herself", "him", "himself", "his", 
"how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", 
"is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", 
"most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", 
"on", "once", "only", "or", "other", "ought", "our", "ours", 
"ourselves", "out", "over", "own", "same", "shant", "she", "shed", 
"shell", "shes", "should", "shouldnt", "so", ""," ", "some", "such", 
"than", "that", "thats", "the", "their", "theirs", "them", "themselves", 
"then", "there", "theres", "these", "they", "theyd", "theyll", 
"theyre", "theyve", "this", "those", "through", "to", "too", 
"under", "until", "up", "very", "was", "wasnt", "we", "wed", 
"well", "were", "weve", "were", "werent", "what", "whats", "when", 
"whens", "where", "\n","\r","wheres", "which", "while", "who", "whos", 
"whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", 
"youd", "youll", "youre", "youve", "your", "yours", "yourself", 
"yourselves", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
"k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", 
"x", "y", "z"]
def generate_ngrams(text,n):

    # split sentences into tokens
    tokens=re.split("\\s+",text)
#     tokens = re.findall(r"\S+", text)
    ngrams=[]

    # collect the n-grams
    for i in range(len(tokens)-n+1):
       temp=[tokens[j] for j in range(i,i+n)]
       ngrams.append(" ".join(temp))

    return ngrams

def remove_stopwords(text):
    text=text.lower()
    text1 = re.sub("[^\w]", " ",  text).split()
    # text1=text.lower().split(" ")
    
    for word in text1:
        if word in sw:
            text1.remove(word)
    return " ".join(text1)

def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans
def remove_nl(text):
    return text.replace('\r', '').replace('\n', '')

def generate_dict(ngrams):
    for i in ngrams:
        if(i in main_dict):
            main_dict[i]=main_dict[i]+1
        else:
            main_dict[i]=1
def vectorize(ngram):
    n_dict={}
    for elem in ngram:
        if(elem not in n_dict):
            n_dict[elem]=1
        else:
            n_dict[elem]=n_dict[elem]+1
    n_vect=[]
    for i in vec:
        if i in n_dict:
            n_vect.append(n_dict[i])
        else:
            n_vect.append(0)
    return n_vect
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


df = pd.read_csv('data.csv')
df['Text']= df['Text'].apply(lambda x:remove_nl(x))
df['news_']= df['Text'].apply(lambda x:remove_punctuation(x))
df['news_sw']= df['news_'].apply(lambda x:remove_stopwords(x))
df['news_sw']= df['news_sw'].apply(lambda x:remove_stopwords(x))
df['news_sw']= df['news_sw'].apply(lambda x:remove_stopwords(x))
df['news_ng']= df['news_sw'].apply(lambda x:generate_ngrams(x.strip(),2))
# df.to_csv("wp.csv")
for i in range(1700):
    generate_dict(df['news_ng'][i])
print(len(main_dict))
# print(main_dict['bond film'])
vec = list(main_dict.keys())
print(len(vec))
df['ngram_vector']= df['news_ng'].apply(lambda x:vectorize(x))
df=df.drop(columns=['news_','news_sw','news_ng','Unnamed: 0','Text'])
print(df.head())
df_test = df.sample(frac=0.2)
df_train = df.drop(df_test.index)
y_train, y_test, X_train, X_test = df_train['Label'], df_test['Label'], df_train.drop(columns=['Label']),df_test.drop(columns=['Label'])
from pt import Perceptron
p=Perceptron()
p.fit(X_train, y_train)
pred=p.predict(X_test)
# from pt import accuracy
import numpy as np
print(accuracy(np.array(y_test),pred))
pred=p.predict(X_train)
print(accuracy(np.array(y_train),pred))

df_test = pd.read_csv('test_data.csv')
df_test['Text']= df_test['Text'].apply(lambda x:remove_nl(x))
df_test['news_']= df_test['Text'].apply(lambda x:remove_punctuation(x))
df_test['news_sw']= df_test['news_'].apply(lambda x:remove_stopwords(x))
df_test['news_sw']= df_test['news_sw'].apply(lambda x:remove_stopwords(x))
df_test['news_sw']= df_test['news_sw'].apply(lambda x:remove_stopwords(x))
df_test['news_ng']= df_test['news_sw'].apply(lambda x:generate_ngrams(x.strip(),2))
df_test['ngram_vector']= df_test['news_ng'].apply(lambda x:vectorize(x))
df_test=df_test.drop(columns=['news_','news_sw','news_ng','Unnamed: 0','Text'])
y_df_test, X_df_test = df_test['Label'],df_test.drop(columns=['Label'])
pred=p.predict(X_df_test)
print(accuracy(np.array(y_df_test),pred))


# a=remove_nl(df['Text'][0])
# a=remove_punctuation(a)
# a=remove_stopwords(a)
# a=remove_stopwords(a)
# a=remove_stopwords(a)
# print(a)
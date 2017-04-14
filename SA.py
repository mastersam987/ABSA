import nltk
import numpy as np
import pandas as pd
import csv
from string import punctuation
import xml.etree.ElementTree as ET
from os import path, access, R_OK
import sys

def main():
    file = sys.argv[1]
    print(file)
    if len(sys.argv) != 2:
        print("Please provide the path to XML file")
        return
    
    sentiment(file)
    return
    
def sentiment(test_file):    
    txt = open(test_file, 'r')
    sent=[]
    for line in txt:
        if '<sentence id' in line:
            line = line.strip()
            line = line.replace('<sentence id="','')
            line = line.replace('">','')  
            sent.append(line)
             
    txt = open(test_file, 'r')
    review=[]
    for line in txt:
    	if '<text>' in line:
    		line = line.replace('</text>', '')
    		line = line.replace('<text>','')
    		line = line.replace('#','')
    		line=line.strip()
    		review.append(line)
    
    revs = review
    rev_pos = []
    for sentence in revs:
        revs_token = nltk.word_tokenize(sentence)
        rev_pos.append(nltk.pos_tag(revs_token))
    
    rev_pos_list = [item for sublist in rev_pos for item in sublist]
    
    noun = ["NN", "NNS", "NNP", "NNPS"]
    
    rev_pos_df = pd.DataFrame.from_records(rev_pos_list, columns = ['Word', 'POS'])
    rev_pos_np = rev_pos_df[rev_pos_df['POS'].isin(noun)] 
    rev_np_count = rev_pos_np['Word'].value_counts().to_frame().reset_index()
    rev_np_count = rev_np_count.rename(columns = {'index':'Word', 'Word':'Count'})
    rev_np_freq = rev_np_count[rev_np_count ['Count'] > 15] 
    
    def strip_punctuation(s):
        return ''.join(c for c in s if c not in punctuation)
    
    def aspectword(row):
        row = strip_punctuation(row)
        sent_split = row.split()
        rev_freq_list = list(rev_np_freq['Word'])
        aspect_terms = set(sent_split).intersection(rev_freq_list)
        aspect_terms_count = rev_np_freq[rev_np_freq['Word'].isin(list(aspect_terms))]
        aspect_term = aspect_terms_count.nlargest(1,'Count').reset_index()
        if aspect_term.shape[0] == 0:
            aspect_final = 'NULL'
        else:
            aspect_final = aspect_term['Word'][0]
        return aspect_final
        
    revs_df = pd.DataFrame(review, columns = ['Sentence'])
    revs_df['Aspect'] = revs_df['Sentence'].apply(aspectword)
    
    negative = []
    with open("words-negative.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            negative.append(row)
    
    positive = []
    with open("words-positive.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            positive.append(row)
    
    def sentiment(text):
        temp = []
        text_sent = nltk.sent_tokenize(text)
        for sent in text_sent:
            n_count = 0
            p_count = 0
            sent_words = nltk.word_tokenize(sent)
            for word in sent_words:
                for item in positive:
                    if(word == item[0]):
                        p_count += 1
                for item in negative:
                    if(word == item[0]):
                        n_count += 1
            if(p_count > 0 and n_count == 0):
                temp.append(1)
            elif(n_count % 2 > 0):
                temp.append(-1)
            elif(n_count % 2 == 0 and n_count > 0):
                temp.append(1)
            else:
                temp.append(0)
        return temp
    
    def polarity(review):
        review = strip_punctuation(review)
        if np.average(sentiment(str(review))) > 0.2:
            result = "Positive"
        elif np.average(sentiment(str(review))) < 0.2:
            result = "Negative"
        else:
            result = "Neutral"
        return result
    
    revs_df['Sentiment'] = revs_df['Sentence'].apply(polarity)
    revs_df['id'] = sent
    
    for row_count in list(range(revs_df.shape[0])):
        print("S"+str(row_count)+": "+revs_df['Sentence'][row_count])
        print("{ \""+revs_df['Aspect'][row_count]+"\", "+revs_df['Sentiment'][row_count]+" }")   
    
    tree = ET.parse(test_file)
    root = tree.getroot()
    list_of_df = []
    for review in root:
        for sentences in review:
            for sentence in sentences:
                for opinions in sentence:
                    for opinion in opinions:
                        list_of_df.append((sentence.attrib, opinion.attrib))
                        
    int_df = pd.DataFrame.from_records(list_of_df, columns = ['id','opinion'])
    
    def id_x(row):
        return list(row.values())[0]
    
    int_df['id'] = int_df['id'].apply(id_x)
    
    def opinion_pol(row):
        polarity = row['polarity']
        return polarity
    
    def opinion_tag(row):
        target = row['target']
        return target
    
    int_df['target'] = int_df['opinion'].apply(opinion_tag)
    int_df['polarity'] = int_df['opinion'].apply(opinion_pol)
    int_df = int_df.drop(labels=['opinion'],axis = 1)
    
    joined  = pd.merge(int_df, revs_df,how = 'left')
    result = joined.rename(columns ={'target':'actual_target','polarity':'actual_polarity','Aspect':'predicted_target','Sentiment':'predicted_polarity'})
    result['predicted_polarity'] = result['predicted_polarity'].str.lower()
    
    def precision(df, actual, predicted):
        num = df[df[actual] == df[predicted]].shape[0] #which are true and we predicted correctly
        den = revs_df.shape[0]
        return num/den
    
    def recall(df, actual, predicted):
        num = df[df[actual] == df[predicted]].shape[0] #which are true and we predicted correctly
        den = df.shape[0]
        return num/den
    
    p_p = precision(result, 'actual_polarity','predicted_polarity')
    r_p = recall(result, 'actual_polarity','predicted_polarity')
    f_measure_p = 2*p_p*r_p/(p_p+r_p)
    
    p_a = precision(result, 'actual_target','predicted_target')
    r_a= recall(result, 'actual_target','predicted_target')
    f_measure_a= 2*p_a*r_a/(p_a+r_a)
    
    print("Aspect Term Evaluation -----------------------------")
    print("Precision:   "+str(p_a))
    print("Recall:      "+str(r_a))
    print("F Measure:   "+str(f_measure_a)+"\n")
    
    print("Polarity Evaluation --------------------------------")
    print("Precision:   "+str(p_p))
    print("Recall:      "+str(r_p))
    print("F Measure:   "+str(f_measure_p))

if __name__ == "__main__":
    main()
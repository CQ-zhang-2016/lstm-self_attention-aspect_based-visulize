import os
import pickle
import json
import numpy as np
from random import choice
from stanfordcorenlp import StanfordCoreNLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
current_path = os.path.dirname(__file__)
# load selectively to save memory and accelerate
def load(load_model=True,load_reviews=True,load_labels=False):
    global entity
    # entity['aspect']['剧情',...] = [enlarged words similar to 剧情,... after my deletion]
    # entity['adj']['正面形容'/'负面形容']
    # entity['verb']['正面动词'/'负面动词']
    # entity['adv']['副词']
    entity = json.load(open(current_path+'\\data\\entity.json','rb'))
    if load_model == True:
        global model
        # Gensim embedding matrix
        model = pickle.load(open(current_path + "\\data\\embedding_matrix3.pickle","rb"))
    if load_reviews == True:
        global reviews,reviews_full
        # some reviews with only divided Chinese words
        # reviews = [sentence1, sentence2,...]
        # sentence = [Chinese_word1, Chinese_word2,...]
        reviews = json.load(open(current_path + "\\data\\reviews_chinese.json","rb"))
        # corresponding some reviews with full sentences
        # reviews_full = [sentence1, sentence2,...]
        # type(sentence) = str
        reviews_full = json.load(open(current_path + "\\data\\reviews_full.json","rb"))
    if load_labels == True:
        global labels
        # dictionary containing potantial tags extracted from reviews
        # appear one time, add one time, has overlap
        # labels['aspect']['剧情',...] = [(potential_tag_for_剧情, corresponding_review_index),...]
        # tag may not in entity
        # labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index),...]
        # tag must in entity['adj']['正面形容'/'负面形容']
        labels = json.load(open(current_path+'\\data\\labels_90000.json','rb'))
# extract potential labels by dependency parsing
def extract_labels(threshold=10000,save=True,loaded=False):
    if loaded == False: load(load_model=False,load_reviews=True,load_labels=False)
    global labels
    nlp = StanfordCoreNLP(r'F:\desktop\web\data\stanford', lang='zh',memory='8g')
    aspect = {'主题':[],'剧情':[],'配乐':[],'画面':[],'节奏':[],'手法':[],'演技':[]}
    sentiment = {'正面':[],'负面':[]}
    arg_aspect = {}
    # arg_aspect[word in entity['aspect'].values()] = word in entity['aspect'].key
    for key,values in entity['aspect'].items():
        for value in values:
            arg_aspect[value] = key
    # extract threshold reviews
    for i in range(threshold):
        # redivide words and analyze part of speeches
        review = ''
        for c in reviews[i]: review += c
        if review == '': continue
        words = nlp.word_tokenize(review)
        part_of_speech = nlp.pos_tag(review)
        # extracting
        try:
            # 1. labels['aspect'] = sentimental adj in entity['adj']
            for speech in part_of_speech:
                if speech[1] == 'VA':
                    if speech[0] in entity['adj']['正面形容']:
                        sentiment['正面'].append((speech[0],i))
                    if speech[0] in entity['adj']['负面形容']:
                        sentiment['负面'].append((speech[0],i))
            # 2. labels['sentiment']['剧情',...] = comments(adj) on 剧情,...
            for dp in nlp.dependency_parse(review):
                if dp[0] == 'nsubj' and words[dp[2]-1] in arg_aspect and part_of_speech[dp[1]-1][1] == 'VA':
                    aspect[arg_aspect[words[dp[2]-1]]].append((words[dp[1]-1],i))
            # 3. output dynamically
            if i % 10 == 0:
                print("Extracting labels:",round(i/threshold*100,2),"%")
        except json.decoder.JSONDecodeError: continue
    # must close JDK to release memory
    nlp.close()
    # save labels
    labels = {'aspect':aspect,'sentiment':sentiment}
    if save == True:
        print("Saving labels_"+str(threshold)+'.json ...')
        with open(current_path+'\\data\\labels_'+str(threshold)+'.json','w') as f:
            json.dump(labels,f)
# train the random-forest classifier on dividing posetive adj and negative adj describing 剧情,...
def trainRF():
    # training samples: entity and adj in labels (subset of entity)
    pos = entity['adj']['正面形容'] + ['好','高','高超','好看']
    neg = entity['adj']['负面形容'] + ['简单']
    for tup in labels['sentiment']['正面']:
        pos.append(tup[0])
    for tup in labels['sentiment']['负面']:
        neg.append(tup[0])
    n = len(pos) + len(neg)
    rf_samples = np.zeros((n,300))
    rf_labels = np.zeros((n,))
    # label it: posetive->1, negative->0
    for i in range(n):
        try:
            if i < len(pos): 
                rf_samples[i] = model.wv[pos[i]]
                rf_labels[i] = 1
            else: 
                rf_samples[i] = model.wv[neg[i-len(pos)]]
                rf_labels[i] = 0
        except KeyError: continue
    # train
    x_train, x_test, y_train, y_test = train_test_split(rf_samples, rf_labels, test_size=0.05, random_state=0)
    rf = RandomForestClassifier(n_estimators = 20)
    rf.fit(x_train, y_train)
    # test
    print("Sentimental classifier accuracy:",round(accuracy_score(y_test, rf.predict(x_test))*100,2),"%")
    return rf
# divide label tuples into positive ones and negative ones
def divide(sample_word,rf):
    n = len(sample_word)
    samples = np.zeros((n,300))
    # transfer word to vectors
    for i in range(n):
        if sample_word[i][0] not in model.wv.index2word: continue
        samples[i] = model.wv[sample_word[i][0]]
    sample_labels = rf.predict(samples)
    # divide label tuples according to their y_pred
    pos = []
    neg = []
    for i in range(n):
        if sample_labels[i] == 1: pos.append(sample_word[i])
        else: neg.append(sample_word[i])
    return [pos,neg]
# given labels, rate the aspects, give tags, recommand similar reviews
def rate_tag_recommand(num_of_tags=10,num_of_rec=30,predicted_score=5,use_lstm=False,loaded=False):
    if loaded == False: load(load_model=True,load_reviews=True,load_labels=True)
    rf = trainRF()
    aspect = labels['aspect']
    sentiment = labels['sentiment']
    score = {}
    # score['剧情',...] = 10*proportion of posetive reviews
    tag = {}
    reviews_total = []
    theme_good = []
    theme_bad = []
    plot_good = []
    plot_bad = []
    sound_good = []
    sound_bad = []
    picture_good = []
    picture_bad = []
    Rhythm_good = []
    Rhythm_bad = []
    technique_good = []
    technique_bad = []
    Acting_good = []
    Acting_bad = []

    # tag['剧情',...] = (<= num_of_tags) of label without tuple
    # analysize each aspect
    for key in aspect:
        review = []
        #print(key)
        # divide sentiment inclination
        aspect[key] = divide(aspect[key],rf)
        # avoid empty list
        if aspect[key][0] == []: aspect[key][0] = [('好',0)]
        if aspect[key][1] == []: aspect[key][1] = [('差',0)]
        # calculate score
        score[key] = round(10*float(len(aspect[key][0]))/(len(aspect[key][0])+len(aspect[key][1])),1)
        # randomly choose tag according to the score
        tag[key] = []
        selected = 0
        repeated = 0
        while selected < num_of_tags:
            # number of posetive labels = score/10
            # proportion of posetive labels = proportion of positive reviews
            if selected < score[key]/10*num_of_tags:
                tag_one = choice(aspect[key][0])[0]
            else: tag_one = choice(aspect[key][1])[0]
            # avoid same tag and avoid death loop
            if tag_one in tag[key]: 
                repeated += 1
                if repeated > 10: selected += 1
                continue
            selected += 1
            tag[key].append(tag_one)
        # recommand relative reviews
        #print("\n")
        #print(key,":",score[key])
        #print("\n标签:",tag[key])
        #print(tag)
        for is_neg in [0,1]:
            review_neg = []
            #if is_neg == 0: print("\n相关正面评论")
            #else:           print("\n相关负面评论")
            printed = []
            for tup in aspect[key][is_neg]:
                if tup[1] in printed: continue
                if len(printed) == num_of_rec: break
                printed.append(tup[1])
                # get reviews from the full reviews file
                #print(len(printed),"\t",reviews_full[tup[1]])
                review_neg.append(reviews_full[tup[1]])
            review.append(review_neg)
        #print(len(review))
        reviews_total.append(review)
        #print(len(reviews_total))
    #reviews_total为7*2*num_of_review
    print(np.array(reviews_total).shape)
    #print(reviews_total[1][0][3])

    # analysize the whole movie
    # if we don't have LSTM to predict the movie's score, calculate it with same method
    if use_lstm == False: predicted_score = round(10*float(len(sentiment['正面']))/(len(sentiment['正面'])+len(sentiment['负面'])),1)
    #print("\n")
    #print("整体 :",predicted_score)
    # randomly choose tag according to the score
    general_tag = []
    selected = 0
    repeated = 0
    while selected < num_of_tags:
        if selected < predicted_score/10*num_of_tags:
            tag_one = choice(sentiment['正面'])[0]
        else: tag_one = choice(sentiment['负面'])[0]
        if tag_one in general_tag: 
            repeated += 1
            if repeated > 10: selected += 1
            continue
        selected += 1
        general_tag.append(tag_one)
    #print("\n标签:",general_tag)
    # recommand relative reviews
    for key in ['正面','负面']:
        #print("\n相关" + key + "评论")
        printed = []
        for tup in sentiment[key]:
            if tup[1] in printed: continue
            if len(printed) == num_of_rec: break
            printed.append(tup[1])
            #print(len(printed),"\t",reviews_full[tup[1]])
    return score,tag,reviews_total
# top design
def main(num_of_tags=10,num_of_rec=30,predicted_score=5,use_lstm=False):
    # load(load_model=True,load_reviews=True,load_labels=False)
    # load model and reviews, labels are extracted below
    extract_labels(threshold=len(reviews),save=True,loaded=True)
    # analysize top threshold reviews
    # save labels
    # do loaded, so don't load again
    rate_tag_recommand(num_of_tags=num_of_tags,num_of_rec=num_of_rec,predicted_score=predicted_score,use_lstm=use_lstm,loaded=True)
    # get num_of_tags for each property
    # recommand num_of_rec relative reviews
    # if use_lstm == True: use predicted_score as the movie rate
    # do loaded, so don't load again
if __name__ == "__main__":
    score,tag,review = rate_tag_recommand()
    #print('now',sentiment)
    print(score)
import json
import os
import pickle
from random import sample
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stanfordcorenlp import StanfordCoreNLP
current_path = os.path.dirname(__file__)

def load(load_model=True,load_reviews=True,load_labels=False):
    """
    Function
        Load model, reviews, labels, entity as global variables
        Load selectively (must load entity) to save memory and accelerate
    Input
        load_model: bool, whether to load model
        load_reviews: bool, whether to load reviews, reviews_full
        load_labels: bool, whether to load labels
    Parameters
        entity: dict
            entity['aspect']['剧情',...] = [enlarged words similar to 剧情,... after my deletion]
            entity['adj']['正面形容'/'负面形容']
            entity['verb']['正面动词'/'负面动词']
            entity['adv']['副词']
        model: gensim object
            Gensim embedding matrix
            Gensim embedding matrix
        reviews: list of list
            some reviews with only divided Chinese words
            reviews = [sentence1, sentence2,...]
            sentence = [Chinese_word1, Chinese_word2,...]
        reviews_full: list of str
            corresponding some reviews with full sentences
            reviews_full = [sentence1, sentence2,...]
            type(sentence) = str
        labels: dict of dict
            dictionary containing potantial tags extracted from reviews
            appear one time, add one time, has overlap
            labels['aspect']['剧情',...] = [(potential_tag_for_剧情, corresponding_review_index),...]
            tag may not in entity
            labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index),...]
            tag must in entity['adj']['正面形容'/'负面形容']
    """
    global entity
    entity = json.load(open(current_path+'\\data\\entity.json','rb'))
    if load_model == True:
        global model
        model = pickle.load(open(current_path + "\\data\\embedding_matrix3.pickle","rb"))
    if load_reviews == True:
        global reviews,reviews_full
        reviews = json.load(open(current_path + "\\data\\reviews_chinese.json","rb"))
        reviews_full = json.load(open(current_path + "\\data\\reviews_full.json","rb"))
    if load_labels == True:
        global labels
        labels = json.load(open(current_path+'\\data\\labels_90000.json','rb'))
def extract_labels(threshold=10000,save=True,loaded=False):
    """
    Function
        Extract potential labels by dependency parsing
        Save labels and set labels as global variables
    Input
        threshold: int, number of reviews to extract labels
        save: bool, whether to save labels
        loaded: bool, if False, load reviews
    Parameters
        labels: dict of dict
            dictionary containing potantial tags extracted from reviews
            appear one time, add one time, has overlap
            labels['aspect']['剧情',...] = [(potential_tag_for_剧情, corresponding_review_index),...]
            tag may not in entity
            labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index),...]
            tag must in entity['adj']['正面形容'/'负面形容']
    """
    if loaded == False: load(load_model=False,load_reviews=True,load_labels=False)
    global labels
    nlp = StanfordCoreNLP(r'D:\数据\大三上\文档\人工智能\Project\stanford-corenlp-full-2018-10-05', lang='zh',memory='8g')
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
            # 1. labels['aspect'] = comments(adj) on 剧情,...
            for speech in part_of_speech:
                if speech[1] == 'VA':
                    if speech[0] in entity['adj']['正面形容']:
                        sentiment['正面'].append((speech[0],i))
                    if speech[0] in entity['adj']['负面形容']:
                        sentiment['负面'].append((speech[0],i))
            # 2. labels['sentiment']['剧情',...] = sentimental adj in entity['adj']
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
        with open(current_path+'\\Record\\labels_'+str(threshold)+'.json','w') as f:
            json.dump(labels,f)
def trainRF():
    """
    Function
        Train the random-forest classifier on dividing posetive adj and negative adj describing 剧情,...
    Output
        rf: sklearn object, trained random forest model
    Parameters:
        pos: list, training and testing samples from entity
        neg: list, training and testing samples from entity
        rf_samples: np.array, word vectors of the samples
        rf_labels: np.array, labels of the samples, 0 -> negative, 1 -> positive
    """
    pos = entity['adj']['正面形容'] + ['好','高','高超','好看', '好棒', '大师级', '没得说', '令人惊叹', '震撼', '完美', '大赞', '没话说', '可贵', '巨牛', '超牛', '深刻', '超棒', '优美', '超赞', '有意思', '难忘', '好赞', '不错', '杠杠','天衣无缝','绝赞', '感动', '一流', '蛮有意思', '牛', '无与伦比', '可牛', '完美无缺', '超一流', '不赖', '真牛', '挺不错', '挺好']
    neg = entity['adj']['负面形容'] + ['简单','二流', '无趣', '劣质', '尬', '无脑', '垃圾', '弱智', '拙劣', '太蠢', '生硬', '弱', '枯燥', '低劣', '粗陋', '辣鸡','白痴','乏味', '太弱', '脑残', '卧槽','心痛','惋惜','不入流', '心碎', '简陋', '苍白', '伤心','智障']
    pos = list(set(pos))
    neg = list(set(neg))
    n = len(pos) + len(neg)
    rf_samples = np.zeros((n,300))
    rf_labels = np.zeros((n,))
    # labelize
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
def divide(sample_word,rf):
    """
    Function
        Divide label tuples into positive ones and negative ones
    Input
        sample_word: list of str, words need to classify
        rf: trained random forest model
    Output
        [pos,neg]: list of list
        pos: list, predicted positive words
        neg: list, predicted negative words
    """
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
def rate_tag_recommand(num_of_tags=10,num_of_rec=30,predicted_score=5,use_lstm=False,loaded=False):
    """
    Function
        Given labels, rate the aspects, give tags, recommand similar reviews
    Input
        num_of_tags: int, number of tags for each aspect
        num_of_rec: int, number of positive/negtive recommanded reviews
        predicted_score: float, if use_lstm == True, use predicted score as the score for '整体'
        use_lstm: bool, whether connect to the LSTM predictor
        loaded: bool, if False, load model, reviews and labels
    Output
        score: dict, predicted score of each aspect, 10 * proportion of posetive reviews
        tag: dict, extracted tags of each aspect, tag['剧情',...] = (<= num_of_tags) of label without tuple
        rec_reviews: dict, rec_reviews['剧情'][0/1] = positive/negative reviews on 剧情
    """
    if loaded == False: load(load_model=True,load_reviews=True,load_labels=True)
    rf = trainRF()
    aspect = labels['aspect']
    aspect['整体'] = [labels['sentiment']['正面'],labels['sentiment']['负面']]
    score = {}
    tag = {}
    rec_review = {}
    # analysize each aspect
    for key in aspect:
        # divide sentiment inclination
        if key != '整体': aspect[key] = divide(aspect[key], rf)
        # avoid empty list
        if aspect[key][0] == []: aspect[key][0] = [('好',0)]
        if aspect[key][1] == []: aspect[key][1] = [('差',0)]
        # calculate score
        score[key] = round(10*float(len(aspect[key][0]))/(len(aspect[key][0]) + len(aspect[key][1])),1)
        # randomly choose tag according to the score
        tag[key] = []
        try:               tagpos = sample(aspect[key][0],int(score[key]/10*num_of_tags))
        except ValueError: tagpos = aspect[key][0]
        try:               tagneg = sample(aspect[key][1],num_of_tags-int(score[key]/10*num_of_tags))
        except ValueError: tagneg = aspect[key][1]
        for element in [tag[0] for tag in tagpos]:
            if element not in tag[key]:
                tag[key].append(element)
        for element in [tag[0] for tag in tagneg]:
            if element not in tag[key]:
                tag[key].append(element)
        # recommand relative reviews
        rec_review[key] = []
        for is_neg in [0,1]:
            try:               review_tup = sample(aspect[key][is_neg],num_of_rec)
            except ValueError: review_tup = aspect[key][is_neg]
            rec_review[key].append([reviews_full[tup[1]] for tup in review_tup])
    # if we don't have LSTM to predict the movie's score, calculate it with same method
    if use_lstm == True: score['整体'] = predicted_score
    return score,tag,rec_review
def visualization(score,tag,rec_review):
    """
    Function
        Print output of rate_tag_recommand properly
    Input (output of rate_tag_recommand)
        score: dict, predicted score of each aspect, 10 * proportion of posetive reviews
        tag: dict, extracted tags of each aspect, tag['剧情',...] = (<= num_of_tags) of label without tuple
        rec_reviews: dict, rec_reviews['剧情'][0/1] = positive/negative reviews on 剧情
    Example
        >>> score,tag,rec_review = rate_tag_recommand()
        >>> visualization(score,tag,rec_review)
        >>> 主题 : 5.4

            标签: ['好', '远', '过硬', '好看', '鲜明', '一般', '够', '差', '幼稚']

            相关正面评论
            1 本来很期待的，这个主题真的很好，可惜导演功力的确不行，展现的东西很肤浅，情节没有
            力度
            ...

            相关负面评论
            1 Netflix的影片质量越来越差了，除了卖个点子之外，成片真的是不忍直视，漏洞随便数数都一大堆；虽然有多达七个主要角色，但千遍一律的性格却极无诚意，对于影片也不是一件好事
            ，对于演员来说则是一个利好。一星半。
            ...
            ...
    """
    for key in score:
        print("\n")
        print(key,":",score[key])
        print("\n标签:",tag[key])
        for is_neg in [0,1]:
            if is_neg == 0: print("\n相关正面评论")
            else:           print("\n相关负面评论")
            count = 0
            for review in rec_review[key][is_neg]:
                count += 1
                print(count,review)
def sb_function_added_by_zpd(sb_review,sb_tag_num=3,sb_rec_num=15,loaded=False):
    """
    Function
        Extract three aspects from one review, and recommand similar reviews.
    Input
        sb_review: str, one user review sentence
        sb_tag_num: int, maximum number of tags extracted from sb_review
        sb_rec_num: int, maximum number of relative reviews recommanded
        loaded: bool, if False, load model and reviews
    Output
        sb_tag: list, tags extracted from the review
        sb_rec_review: list, similar recommanded reviews
    Example
        >>> sb_tag,sb_rec_review = sb_function_added_by_zpd('性格差，剧情奇怪，主题不行')
        >>> print(sb_tag)
        >>> ['性格','剧情','主题']
        
        >>> print(sb_rec_review)
        >>> ['完全就是黄渤带着一帮朋友把十几个小品塞到一个大纲里，基本可以看出电影是边拍边修改剧本的，
        黄渤同时在导演身份上引用了太多自己出身演员的经验，整个作品都显得太信手拈来，对“末世”主题的理解太肤浅，太多临场发挥的尴尬台词，角色深度不够，
        剧情转折莫名其妙，感情线的黑人问号脸等。但最让我受不了的就是角色和演员本身性格太过融合，一度觉得这是在看真人秀……]
    """
    if loaded == False: load(load_model=False,load_reviews=True,load_labels=True)
    nlp = StanfordCoreNLP(current_path + '\\data\\stanford-corenlp-full-2018-10-05', lang='zh',memory='8g')
    words = nlp.word_tokenize(sb_review)
    sb_tag = []
    count = 0
    # extract nones of all the nsubj
    for dp in nlp.dependency_parse(sb_review):
        if count == sb_tag_num: break
        if dp[0] == 'nsubj' and words[dp[2]-1] not in sb_tag: 
            sb_tag.append(words[dp[2]-1])
            count += 1
    nlp.close()
    # count the tag appearance in all the review and sort
    sb_times = {}
    for i in range(len(reviews)):
        sb_times[i] = 0
        for word in sb_tag:
            if word in reviews[i]:
                sb_times[i] += 1
    sb_times = sorted(sb_times.items(), key=lambda d:d[1],reverse=True)
    # recommand reviews with tags appearing most times
    sb_rec_review = []
    count = 0
    for review_tup in sb_times:
        count += 1
        if count == sb_rec_num or review_tup[1] == 0: break
        sb_rec_review.append(reviews_full[review_tup[0]])
    print(len(reviews_full))
    return sb_tag,sb_rec_review
def main(num_of_tags=10,num_of_rec=30,predicted_score=5,use_lstm=False):
    """
    Function
        Demonstrate the top design of the whole procedure
    Procedure
        load
            load model and reviews, labels are extracted below
        extract_labels
            analysize top threshold reviews
            save labels
            do loaded, so don't load again
        rate_tag_recommand
            get num_of_tags tags for each property
            recommand num_of_rec relative reviews
            if use_lstm == True: use predicted_score as the movie rate
            do loaded, so don't load again
                trainRF: rf = trainRF()
                divide:  aspect[key] = divide(aspect[key],rf)
        recommand reviews in same aspects
    """
    load(load_model=True,load_reviews=True,load_labels=False)
    extract_labels(threshold=len(reviews),save=True,loaded=True)
    score,tag,rec_review = rate_tag_recommand(num_of_tags=num_of_tags,num_of_rec=num_of_rec,predicted_score=predicted_score,use_lstm=use_lstm,loaded=True)
    visualization(score,tag,rec_review)
    sb_tag,sb_rec_review = sb_function_added_by_zpd('性格差，剧情奇怪，主题不行')
    print(sb_tag)
    print(sb_rec_review)
if __name__ == "__main__":
    sb_tag,sb_rec_review = sb_function_added_by_zpd('导演说，拍这部电影是想让现在的年轻人知道，发哥意味着什么——还用问嘛，意味着天下无双的帅啊！！'
                                                    '怀揣艺术匠心开创假钞帝国的大亨，乍听有点像奥斯卡路数的枭雄传记片，影片整体那种复古质感也很好。'
                                                    '片中周润发多次小马哥附体，颜值，大长腿，最重要的是又雅又痞，风度翩翩，一比这届鲜肉真是弱爆了。郭天王也很好玩，'
                                                    '这次演个怂包（伪）。发哥和城城真是强攻&弱受组合'
                                                    '，对话完全是个喜剧片（当然两位演技都很棒，搞笑也能立住不垮），看到最后又一切都能说通了，'
                                                    '也到最后才知道片名是什么含义。今年国庆档三强，无双、影、李茶的姑妈，居然都是讲一个人的真假身份，有趣。')
    print(sb_tag)
    print(sb_rec_review)
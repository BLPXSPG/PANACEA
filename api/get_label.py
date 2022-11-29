import os
import json
import math
import shutil
import urllib.parse
import requests
import string
import csv
from rank_bm25 import BM25Okapi
import pickle
import os
from absl import app, flags

flags.DEFINE_string("run",
                    default="label",
                    help="Model used.")
flags.DEFINE_string("embedding",
                    default="bert",
                    help="Embedding model used.")
flags.DEFINE_boolean("123",
                    default=True,
                    help="")

FLAGS = flags.FLAGS



project_dir = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(project_dir, "data", "tweets")
unlabelled_data_path = os.path.join(project_dir, "data", "tweet_forest")
"""label_dir = os.path.join(project_dir, "data", "COVID-RV.csv")
info_dic = {}
claims = []
with open(label_dir, newline='') as csvfile:  
    spamreader = csv.reader(csvfile, delimiter=',')
    next(spamreader)
    for row in spamreader:
        info_dic[row[1]] = [row[0], row[2]]
        claims.append(row[0])
    csvfile.close()
claims = list(set(claims))"""


def get_label():
    from debater_python_api.api.debater_api import DebaterApi
    debater_api = DebaterApi('90c57c807cfd41ad1eb996042e1039ceL05')
    #from nltk.sentiment.vader import SentimentIntensityAnalyzer
    #sid = SentimentIntensityAnalyzer()
    pro_con_client = debater_api.get_pro_con_client()
    import re
    import tweepy
    
    consumer_key = 'osMf5J0IgRnC2ss1f9oqSl08a'
    consumer_secret = 'OUN9dEn2krD6RwUT3H2HEjzaM5q8crfJejLoXl1pOowZGoIVrN'
    access_token = '1067724593178308608-KAYiMkpgjEmd7z0lYP9Th1ATDqzh1t'
    access_secret = 'IP6eeIcld5OkB8MvEjVdmgbhHFU9Orjz5A7tZQK2XYuDe'
    tweetsPerQry = 2
    maxTweets = 6
    auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_secret
    )
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    def getstance(claim, text):
        #sentence_topic_dicts = [{'sentence' : text, 'topic' : claim.claim }]
        sentence_topic_dicts = [{'sentence' : text, 'topic' : claim }]
        stance_score = pro_con_client.run(sentence_topic_dicts)[0]
        #sent_score = sid.polarity_scores(text)['compound']
        sent_score = stance_score
        return stance_score, sent_score
    
    def get_tweetcontent(source_tweet_id, tweet_root_dir):
        try:
            source_tweet = api.get_status(source_tweet_id, tweet_mode="extended")._json 
            #retweet_list = get_retweets(source_tweet_id, source_tweet['retweet_count'])
            print("collected", source_tweet_id, source_tweet['full_text'], source_tweet['retweet_count'])
            print()
            location, coordinates = getlocation(source_tweet)
            rephrase_tweet = {"id": source_tweet_id, "text": source_tweet['full_text'], "created_at": source_tweet["created_at"],  "public_metrics": {"retweet_count": source_tweet['retweet_count']}, "location": location, "coordinates": coordinates}
            with open(tweet_root_dir, "w") as f:
                json.dump(rephrase_tweet, f)
                f.close()
        except Exception as e: 
            # if source tweet is deleted, then we discard the whole spread tree
            print(e, "source tweet deleted", source_tweet_id)
    
    def getlocation(tweet):
        #input location, output longitude and latitude
        if not isinstance(tweet['coordinates'], type(None)):
            return tweet['location'], tweet['coordinates']
        if not isinstance(tweet['geo'], type(None)):
            return tweet['location'], tweet['geo']['coordinates']
        if not isinstance(tweet['place'], type(None)):
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(tweet['place']['name']) + '?format=json'
            response = requests.get(url).json()
            if len(response) != 0:
                return tweet['user']['location'], [response[0]['lon'], response[0]['lat']]
            else:
                return 'none', [0, 0]
        if tweet['user']['geo_enabled']:
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(tweet['user']['location']) + '?format=json'
            response = requests.get(url).json()
            #print("response", tweet['user']['location'], response)
            if len(response) != 0:
                return tweet['user']['location'], [response[0]['lon'], response[0]['lat']]
            else:
                return 'none', [0, 0]
        return 'none', [0, 0]
    
    def resort(split_file, claim, tweet, tweet_id, label, agreement):
        with open(split_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            for row in spamreader:
                if row == []:
                    continue
                if row[3] not in tweet_id:
                    claim.append(row[1])
                    tweet_text = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",row[2]).split()).rstrip()
                    tweet.append(tweet_text)
                    tweet_id.append(row[3])
                    label.append(row[4])
                    agreement.append(row[5])
                else:
                    print("duplicate", row[3])
        return claim, tweet, tweet_id, label, agreement

    label_dir = os.path.join(project_dir, "data", "Data_E", "COVID-RV.csv")
    if not os.path.isfile(label_dir):
        claim = []
        tweet = []
        tweet_id = []
        label = []
        agreement = []
        claim, tweet, tweet_id, label, agreement = resort(os.path.join(project_dir, "data", "Data_E", "a1.csv"), claim, tweet, tweet_id, label, agreement)
        claim, tweet, tweet_id, label, agreement = resort(os.path.join(project_dir, "data", "Data_E", "a2.csv"), claim, tweet, tweet_id, label, agreement)
        split_file = os.path.join(project_dir, "data", "Data_E", "a3.csv")
        with open(split_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            for row in spamreader:
                if row[2] not in tweet_id:
                    claim.append(row[0])
                    tweet_text = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",row[1]).split()).rstrip()
                    tweet.append(tweet_text)
                    tweet_id.append(row[2])
                    label.append(row[3])
                    agreement.append(row[4])
                else:
                    print("duplicate", row[2])
        with open(label_dir, 'w') as file:
            writer = csv.writer(file)
            for i in range(len(claim)):
                writer.writerow([claim[i], tweet[i], tweet_id[i], label[i], agreement[i]])
            file.close()
    
    with open(label_dir, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        #claim[i], tweet[i], tweet_id[i], label[i], agreement[i]
        comment_path = os.path.join(project_dir, "data", "Data_E", "conversations")
        tweet_save_dir = os.path.join(project_dir, "data", "Data_E", "conversations_withlabel")
        tweet_root_dir = os.path.join(project_dir, "data", "Data_E", "conversations_root")
        if not os.path.isdir(tweet_save_dir):
            os.makedirs(tweet_save_dir)
        if not os.path.isdir(tweet_root_dir):
            os.makedirs(tweet_root_dir)

        for row in spamreader:
            comment_tweet = row[2] + ".jsonl"
            #for comment_tweet in os.listdir(comment_path):
            tweet_source_path = os.path.join(comment_path, comment_tweet)
            # crawl source tweet:
            tweet_root_path = os.path.join(tweet_root_dir, comment_tweet)
            if not os.path.isfile(tweet_root_path):
                get_tweetcontent(row[2], tweet_root_path)
                
            # load comment tweet
            if os.path.isfile(os.path.join(tweet_save_dir, comment_tweet)):
                tweet_source_path = os.path.join(tweet_save_dir, comment_tweet)
            elif os.path.isfile(os.path.join(comment_path, comment_tweet)):
                tweet_source_path = os.path.join(comment_path, comment_tweet)
            else:
                continue
            with open(tweet_source_path, "r") as f:
                data = f.readlines()
                f.close()
            tree_info = []
            for tweet in data:
                tweet = json.loads(tweet)
                text = tweet['text']
                claim = row[0]
                text = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()).rstrip()

                if "stance_score" not in tweet:
                    if len(text) < 5:
                        stance_score, sent_score = 0, 0
                        print(claim, text)
                    else:
                        stance_score, sent_score = getstance(claim, text)
                    tweet["stance_score"] = stance_score
                    tweet["sent_score"] = sent_score

                if "location" not in tweet:
                    try:
                        source_tweet = api.get_status(tweet['id'], tweet_mode="extended")._json 
                        location, coordinates = getlocation(source_tweet)
                        tweet["location"] = location
                        tweet["coordinates"] = coordinates
                    except Exception as e: 
                        # if source tweet is deleted, then we discard the whole spread tree
                        print(e, "source tweet deleted", tweet['id'])
                        tweet["location"] = "none"
                        
                tree_info.append(tweet)
            # save tweet with label
            with open(os.path.join(tweet_save_dir, comment_tweet), "w") as outfile:
                outfile.close()
                    
            with open(os.path.join(tweet_save_dir, comment_tweet), "a") as outfile:
                for tweet in tree_info:
                    json.dump(tweet, outfile)
                    outfile.write('\n')
                outfile.close()


def get_topic(num_topics=10):
    from sklearn.decomposition import PCA
    from scipy.spatial import distance
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer 
    import numpy as np
    import torch as T
    lemmatizer = WordNetLemmatizer()
    
    def load_glove(glove_path):
        embeddings_dict = {}
        with open(glove_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict
    
    def topicmodel(word_list, num_topics):
        import gensim
        import gensim.corpora as corpora
        # Create Dictionary
        id2word = corpora.Dictionary(word_list)
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in word_list]
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
        return lda_model
    
    def get_sentence_embeddings(sentence, embedding_dic):
        sentence_embedding = 0
        for word in sentence:
            if word in embedding_dic:
                try:
                    sentence_embedding += embedding_dic[word]
                except:
                    sentence_embedding = embedding_dic[word]
            else:
                print(word, "not in glove")
        return sentence_embedding/len(sentence)
    
    def find_closest_embeddings(embeddings_dict, embedding, distance_type):
        if distance_type == "euclidean":
            return sorted(embeddings_dict.keys(), key=lambda word: distance.euclidean(embeddings_dict[word], embedding))
        elif distance_type == "cosine":
            return sorted(embeddings_dict.keys(), key=lambda word: distance.cosine(embeddings_dict[word], embedding))
        
    def bert_embedding(sentences_list):
        from transformers import BertForPreTraining, BertTokenizer
        model = BertForPreTraining.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
        tokenized_sent = tokenizer(sentences_list, return_tensors="pt", padding=True)
        outputs = model(**tokenized_sent)
        #representation = T.mean(outputs.prediction_logits, 1)
        last_hidden_state = outputs.prediction_logits[:,0,:]
        #sentences_list = ["test embedding", "hello world"]
        return last_hidden_state.detach().numpy()
   
    def get_topic_embedding(top_words, embedding_dic):
        reweight = 0
        if FLAGS.embedding == "glove":
            for word in top_words:
                if word[0] in embedding_dic:
                    reweight += word[1]
                    try:
                        topic_embedding += word[1]*embedding_dic[word[0]]
                    except:
                        topic_embedding = word[1]*embedding_dic[word[0]]
                else:
                    print(word, "not in glove")
        elif FLAGS.embedding == "bert":
            word_list = [word[0] for word in top_words]
            word_embeddings = bert_embedding(word_list)
            for i, word in enumerate(top_words):
                reweight += word[1]
                try:
                    topic_embedding += word[1]*word_embeddings[i]
                except:
                    topic_embedding = word[1]*word_embeddings[i]
        return topic_embedding, reweight
    
    def get_embedding(claim_input):
        print("claim", claim_input)
        label_dir = os.path.join(project_dir, "data", "Data_E", "COVID-RV.csv")
        doc_list = []
        sentence_embedding_dic = {}
        sentence_texts= []
        with open(label_dir, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            #claim[i], tweet[i], tweet_id[i], label[i], agreement[i]
            comment_path = os.path.join(project_dir, "data", "Data_E", "conversations")
            tweet_save_dir = os.path.join(project_dir, "data", "Data_E", "conversations_withlabel")
            for row in spamreader:
                claim = row[0].lower()
                if claim != claim_input:
                    continue
                comment_tweet = row[2] + ".jsonl"
                
                if os.path.isfile(os.path.join(tweet_save_dir, comment_tweet)):
                    tweet_source_path = os.path.join(tweet_save_dir, comment_tweet)
                elif os.path.isfile(os.path.join(comment_path, comment_tweet)):
                    tweet_source_path = os.path.join(comment_path, comment_tweet)
                else:
                    continue
                with open(tweet_source_path, "r") as f:
                    data = f.readlines()
                    f.close()
                for tweet in data:
                    tweet = json.loads(tweet)
                    tweet_text = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet['text']).split()).rstrip()
                    tweet_text = tweet_text.lower()
                    word_list = tweet_text.translate(str.maketrans('','',string.punctuation))
                    word_list = word_tokenize(word_list)
                    word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words('english')]
                    if len(word_list) < 2:
                        continue
                    if FLAGS.embedding == "glove":
                        sentence_embedding_dic[tweet_text] = word_list
                    doc_list.append(word_list)
                    sentence_texts.append(tweet_text)
        
            if FLAGS.embedding == "glove":
                return doc_list, sentence_embedding_dic
            elif FLAGS.embedding == "bert":
                return doc_list, sentence_texts
                    
                    
    topic_dir = os.path.join(project_dir, "data", "Data_E", "topic_info.json")
    claim_path = os.path.join(project_dir, "data", "Data_E", "claims.json")
    if not os.path.isfile(claim_path):
        label_dir = os.path.join(project_dir, "data", "Data_E", "COVID-RV.csv")
        claims = []
        with open(label_dir, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                claims.append(row[0].lower())
        claims = list(set(claims))
        with open(claim_path, "w") as outfile:
            json.dump({"claim": claims}, outfile)
            outfile.close()
    else:
        with open(claim_path, "r") as outfile:
            claims = json.load(outfile)["claim"]
            outfile.close()
        
    
    if FLAGS.embedding == "glove":
        glove_path = os.path.join(project_dir, "data", "glove.6B.200d.txt")
        embedding_dic = load_glove(glove_path)
    else:
        embedding_dic = {}
    pca = PCA(n_components=2)
    topic_to_save = {}
    for claim in claims:
        #get all sentence embedding for each claim
        doc_list, sentences = get_embedding(claim) 
        sentence_embedding_dic = {}
        #get topics
        print("train lda_model")
        lda_model = topicmodel(doc_list, num_topics)
        
        print("start build sentence_embedding_dic")
        if FLAGS.embedding == "glove":
            for tweet_text in sentences:
                sentence_embedding_dic[tweet_text] = get_sentence_embeddings(sentences[tweet_text], embedding_dic)
        elif FLAGS.embedding == "bert":
            batch_size = 64
            n_batches = int(len(sentences)//batch_size - 1)
            #embedding_all = []
            for i in range(n_batches):
                sentence_picked = sentences[i*batch_size: (i+1)*batch_size]
                embedding_all = bert_embedding(sentence_picked)
                #print(embedding_all.shape())
                for j in range(batch_size):
                    sentence_embedding_dic[sentence_picked[j]] = embedding_all[j]
                print("bert embedding, batch", i, n_batches, embedding_all[j], len(list(sentence_embedding_dic.keys())))
            sentence_picked = sentences[(i+1)*batch_size:]
            embedding_all = bert_embedding(sentence_picked)
            for j in range(len(sentence_picked)):
                sentence_embedding_dic[sentence_picked[j]] = embedding_all[j]
            #embedding_all.append(bert_embedding(sentence_picked), model, tokenizer)
            #sentence_embedding_dic[claim] = T.cat(embedding_all, 0)
        
        from numpy import inf
        print("start finding topic coordinates")
        topic_embeddings = []
        topic_sentences = []
        topic_weights = []
        topic_words = []
        for i in range(num_topics):
            top_words = lda_model.show_topic(i)
            topic_embedding, reweight = get_topic_embedding(top_words, embedding_dic)
            topic_embedding[~np.isfinite(topic_embedding)] = 0
            np.nan_to_num(topic_embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            print(topic_embedding)
            topic_embeddings.append(topic_embedding)
            topic_sentences.append(find_closest_embeddings(sentence_embedding_dic, topic_embedding, "euclidean")[0])
            topic_weights.append(reweight)
            top_words = [[str(word[0]), str(word[1])] for word in top_words]
            topic_words.append(top_words)
        print("PCA fit")
        try:
            pca.fit(topic_embeddings)
        except:
            print(topic_embeddings)
        topic_embeddings = pca.transform(topic_embeddings).tolist()
        topic_embeddings = [[str(coor[0]), str(coor[1])] for coor in topic_embeddings]
        print(claim)
        print(topic_sentences)
        print(topic_embeddings)
        print(topic_weights)
        total_weights = sum(topic_weights)
        topic_weights = [str(weight/total_weights) for weight in topic_weights]
        print(topic_words)
        print(type(claim), type(topic_embeddings), type(topic_weights), type(topic_sentences), type(topic_words))
        # format: {"claim 1": {"coordinates": [x, y, weight], "top_words": [[word, weight], [word, weight]]}}
        topic_to_save[claim] = {"coordinates": topic_embeddings, "weight": topic_weights, "top_sentence": topic_sentences, "top_words": topic_words}
          
    #save topic information 
    topic_dir = os.path.join(project_dir, "data", "Data_E", "topic_info.json")
    with open(topic_dir, "w") as outfile:
        json.dump(topic_to_save, outfile)
        outfile.close()
    
    
    """
    for i in range(num_topics):
        top_words = lda_model.show_topic(i)
        #x, y, top_sentence = self.get_topic_info(top_words, embedding_dic, sentence_embedding_dic)
        topic = Topic(x=pca.singular_values_[i][0], y=pca.singular_values_[i][1], text=topic_sentences[i])
        for word in top_words:
            word = TopicWords(topic=topic, word=word[0], weight=word[1])
    print("topic words saved")
    pass"""


def get_label_old():
    from debater_python_api.api.debater_api import DebaterApi
    debater_api = DebaterApi('90c57c807cfd41ad1eb996042e1039ceL05')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    pro_con_client = debater_api.get_pro_con_client()
    
    
    
    def getstance(claim, text):
        #sentence_topic_dicts = [{'sentence' : text, 'topic' : claim.claim }]
        sentence_topic_dicts = [{'sentence' : text, 'topic' : claim }]
        stance_score = pro_con_client.run(sentence_topic_dicts)[0]
        sent_score = sid.polarity_scores(text)['compound']
        return stance_score, sent_score
    
    for tweet_folder in os.listdir(data_path):
        comment_path = os.path.join(data_path, tweet_folder, "comment")
        tweet_save_dir = os.path.join(data_path, tweet_folder, "comment_withlabel")
        if not os.path.isdir(tweet_save_dir):
            os.makedirs(tweet_save_dir)
        for comment_tweet in os.listdir(comment_path):
            tweet_source_path = os.path.join(data_path, tweet_folder, "comment", comment_tweet)
            # load tweet
            with open(tweet_source_path, "r") as f:
                tweet = json.load(f)
                text = tweet['full_text']
                claim = info_dic[tweet_folder][0]
                stance_score, sent_score = getstance(claim, text)
                tweet["stance_score"] = stance_score
                tweet["sent_score"] = sent_score
                f.close()
            # save tweet with label
            with open(os.path.join(tweet_save_dir, comment_tweet), "w") as f:
                json.dump(tweet, f)
                f.close()

def get_vocab(corpus):
    import nltk
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer 
    
    lemmatizer = WordNetLemmatizer()
    word_list = []
    for tweet in corpus:
        tweet = tweet.translate(str.maketrans('','',string.punctuation))
        tweet = tweet.lower()
        word_list += [lemmatizer.lemmatize(word) for word in word_tokenize(tweet) if word not in stopwords.words('english')]
    
    # get word dic for word could
    word_dic = {}
    for i in word_list:
        word_dic[i] = word_dic.get(i, 0) + 1
    vocab = sorted(word_dic, key=word_dic.get, reverse=True)[:5000]
    vocab_dic = {}
    for i, word in enumerate(vocab):
        vocab_dic[word] = i
    vocab_save_path = os.path.join(project_dir, "data", "vocab.json")
    with open(vocab_save_path, "w") as f:
        json.dump(vocab_dic, f)
        f.close()
    
    # get topics
    topicmodel(word_list)
    


def topicmodel(word_list, num_topics = 10):
    import gensim
    from gensim.utils import simple_preprocess
    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(word_list)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in word_list]
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
    
    topics = lda_model.print_topics()
    print(topics)
    return topics

        
def trainbm25(labelled_data=True, unlabelled_data=True):
    corpus = []
    ids = []
    if labelled_data:
        for tweet_folder in os.listdir(data_path):
            tweet_source_path = os.path.join(data_path, tweet_folder, "source-tweets", tweet_folder+".json")
            with open(tweet_source_path, "r") as f:
                tweet = json.load(f)
                corpus.append(tweet['full_text'])
                f.close()
            comment_path = os.path.join(data_path, tweet_folder, "comment")
            for comment_tweet in os.listdir(comment_path):
                tweet_source_path = os.path.join(data_path, tweet_folder, "comment", comment_tweet)
                # load tweet
                with open(tweet_source_path, "r") as f:
                    tweet = json.load(f)
                    corpus.append(tweet['full_text'])
                    ids.append(tweet['id_str'])
                    f.close()
    if unlabelled_data:
        for tweet_folder in os.listdir(unlabelled_data_path):
            for tweet_id in os.listdir(comment_path):
                comment_path = os.path.join(unlabelled_data_path, tweet_folder, tweet_id)
                # load tweet
                with open(comment_path, "r") as f:
                    tweet = json.load(f)
                    corpus.append(tweet['text'])
                    ids.append(tweet['id_str'])
                    f.close()

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    get_vocab(corpus)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_save_path = os.path.join(project_dir, "data", "bm25result")
    with open(bm25_save_path, 'wb') as bm25result_file:
        pickle.dump(bm25, bm25result_file)
        

def bow(text):
    return text



def rumourdetectionclean(labelled_data=True, unlabelled_data=True):
    # vocab
    vocab_save_path = os.path.join(project_dir, "data", "vocab.json")
    with open(vocab_save_path, "w") as f:
        vocab = json.load(f)
        f.close()
    
    if labelled_data:
        for tweet_folder in os.listdir(data_path):
            tree_info = {}
            child_count = {}
            tweet_source_path = os.path.join(data_path, tweet_folder, "source-tweets", tweet_folder+".json")
            with open(tweet_source_path, "r") as f:
                tweet = json.load(f)
                id = tweet['id_str']
                parent_id = tweet['source_tweet']
                bow = bow(tweet['full_text'])
                tree_info[id] = [parent_id, bow]
                if parent_id in child_count:
                    child_count[parent_id] += [id]
                else:
                    child_count[parent_id] = [id]
                f.close()
            comment_path = os.path.join(data_path, tweet_folder, "comment")
            for comment_tweet in os.listdir(comment_path):
                tweet_source_path = os.path.join(data_path, tweet_folder, "comment", comment_tweet)
                with open(tweet_source_path, "r") as f:
                    tweet = json.load(f)
                    id = tweet['id_str']
                    parent_id = tweet['source_tweet']
                    bow = bow(tweet['full_text'])
                    tree_info[id] = [parent_id, bow]
                    if parent_id in child_count:
                        child_count[parent_id] += [id]
                    else:
                        child_count[parent_id] = [id]
                    f.close()
    if unlabelled_data:
        for tweet_folder in os.listdir(unlabelled_data_path):
            for tweet_id in os.listdir(comment_path):
                comment_path = os.path.join(unlabelled_data_path, tweet_folder, tweet_id)
                # load tweet
                with open(comment_path, "r") as f:
                    tweet = json.load(f)
                    corpus.append(tweet['text'])
                    ids.append(tweet['id_str'])
                    f.close()
        
        
        
def main(argv):
    if FLAGS.run == "label":
        get_label()
    elif FLAGS.run == "topic":
        get_topic()

if __name__ == "__main__":
    app.run(main)
# -*- coding: utf8 -*-

from dis import show_code
from unicodedata import category
from unittest import result
from django.shortcuts import render
from rest_framework import generics, status
from .serializers import DocumentRowSerializer, SentenceRowSerializer, SentenceSerializer, DocumentDetailsSerializer, TweetCountSerializer
from .serializers import VocabularySerializer, TwitterSerializer, TweetSpreadSerializer, TweetMapSerializer, TweetGraphSerializer, ClaimSerializer, TopicSerializer
from .models import Document, Sentence, TweetCount, TweetSpread, Twitter, Vocabulary, Claim, TweetMap, TweetGraph, Topic, TopicWords
from .search_model import SearchModel
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse



from rest_framework.throttling import UserRateThrottle


# Using session outside view
# from django.contrib.sessions.backends.db import SessionStore

#import torch
#import torch.nn as nn
import os
import json
import math
import shutil
import urllib.parse
import requests
import string
import en_core_web_sm
import csv
nlp = en_core_web_sm.load()
model = SearchModel()
import pickle

with open(os.path.join(os.path.dirname(__file__), 'bm25result'), 'rb') as bm25result_file:
    bm25result = pickle.load(bm25result_file)
project_dir = os.path.abspath(os.path.dirname(__file__))
# REST_FRAMEWORK = {
#     # 'DEFAULT_RENDERER_CLASSES': [
#     #     'rest_framework.renderers.JSONRenderer',
#     # ],
#     # 'DEFAULT_PARSER_CLASSES': [
#     #     'rest_framework.parsers.JSONParser',
#     # ]
#     'DEFAULT_THROTTLE_CLASSES': [
#         'rest_framework.throttling.AnonRateThrottle',
#         'rest_framework.throttling.UserRateThrottle'
#     ],
#     'DEFAULT_THROTTLE_RATES': {
#         'anon': '10/second',
#         'user': '100/second'
#     }
# }

from rest_framework.settings import api_settings

print('wawawawawawa!', api_settings.DEFAULT_THROTTLE_RATES['anon'])


class GetDocument(APIView):
    def get(self, request, format=None):
        #documentset = Document.objects.all()
        documentset = Document.objects.filter(host=self.request.session.session_key)
        serializer_class = DocumentRowSerializer(documentset, many=True)
        #print(serializer_class.data)
        return Response(serializer_class.data)


class GetDocumentDetails(APIView):
    lookup_url_kwarg = 'id'
    def get(self, request, format=None):
        id = request.GET.get(self.lookup_url_kwarg)
        documentset = Document.objects.filter(id=id)
        serializer_class = DocumentDetailsSerializer(documentset, many=True)
        """document = Document.objects.get(id=id)
        sentenceset = document.sentences.all()
        #sentenceset = Sentence.objects.all()
        serializer_class = SentenceSerializer(sentenceset, many=True)"""
        return Response(serializer_class.data)


class GetSentence(APIView):
    def get(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        sentenceset = Sentence.objects.filter(host=self.request.session.session_key)
        serializer_class = SentenceRowSerializer(sentenceset, many=True)
        return Response(serializer_class.data)

def match_stance(sent_list, sent_info):
    sent_stance = {}
    for sentence_id in sent_info:
        sentence = sent_info[sentence_id]
        sent_stance[sentence["sent"]] = [sentence["sent_similarity"], sentence["sent_inference"]]
    
    output_similarity = []
    output_stance = []
    for sentence in sent_list:
        if sentence in sent_stance:
            output_similarity.append(sent_stance[sentence][0])
            output_stance.append(sent_stance[sentence][1])
        else:
            output_similarity.append(0)
            output_stance.append([0, 0, 0])
    return json.dumps(output_similarity), json.dumps(output_stance)


def search_model(query):
    """saved_model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nli-san_simpl4vp1.pt")
    model_loaded = torch.load(saved_model_file)"""
    print("start search documents++++++++++")

    full_results_df = model.update_query(query)
    results = full_results_df.loc[0, 'results']
    veracity = full_results_df.loc[0, 'veracity']
    #print(full_results_df.loc[0,'query'])

    """data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testoutput.json") 
    with open(data_path, "r") as f:
        data = json.load(f)
        f.close()
    results = data['results']
    veracity = data['veracity']"""
    return results, veracity

    import random

    output = {}
    stance_list = ["Negative", "Neutral", "Positive"]
    for document_id in results:
        document = results[document_id]
        document_info = {}
        document_info["title"] = document["title"]
        document_info["abstract"] = document["content"][:300]
        document_info["content"] = document["content"]
        sent_list = sent_tokenize(document["content"])
        document_info["sent_list"] = json.dumps(sent_list)
        document_info["sent_similarity"], document_info["sent_stance"] = match_stance(sent_list, document["sents"])
        doc_sentiment = document["content_inference"]
        document_info["sentiment"] = doc_sentiment
        document_info["stance"] = stance_list[doc_sentiment.index(max(doc_sentiment))]
        document_info["url"] = document["url"]
        document_info["source"] = document["source"]
        sentence_info_list = []
        for sentence_id in document["sents"]:
            sentence = document["sents"][sentence_id]
            sentence_info = {}
            sentence_info["sentence"] = sentence["sent"]
            sentence_info["content"] = sentence["sent_context"]
            sent_sentiment = sentence["sent_inference"]
            sentence_info["sentiment"] = sent_sentiment
            sentence_info["similarity"] = sentence["sent_similarity"]
            sentence_info["stance"] = stance_list[sent_sentiment.index(max(sent_sentiment))]
            sentence_info_list.append(sentence_info)
        document_info["sentence"] = sentence_info_list
        output[document_id] = document_info
    print("finished")
    return output, veracity


class InitQuery(APIView):
    def initial(self, request):
        if self.request.session.exists(self.request.session.session_key):
            print('00000000000 Initialize session manager')
            if len(self.request.session.keys()) != 0:
                self.request.session.flush()
                self.request.session.clear()


class checkQueueState(APIView):
    def get(self, request):
        x_forwarded_for = self.request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            user_ip = x_forwarded_for.split(',')[0]
        else:
            user_ip = self.request.META.get('REMOTE_ADDR')
        # The session has registered
        print('The keys are', self.request.session.keys())
        if user_ip in self.request.session:
            print('xxxxxxxxxxxxxThe user has already made a request and is being processed')
            self.request.session.modified = True
            self.request.session.save()
            data = {"session_reply": 0}
            return JsonResponse(data, status=status.HTTP_200_OK)
        elif len(self.request.session.keys()) != 0:
            print('xxxxxxxxxxxxxThe user hasn\'t made a request, and the GPUs are busy')
            data = {"session_reply": len(self.request.session.keys())}
            return JsonResponse(data, status=status.HTTP_200_OK)
        else:
            print('xxxxxxxxxxxxxRequest done')
            data = {"session_reply": -1}
            return JsonResponse(data, status=status.HTTP_200_OK)

class InputQuery(APIView):
    throttle_classes = [UserRateThrottle]

    def post(self, request, format=None):
        x_forwarded_for = self.request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            user_ip = x_forwarded_for.split(',')[0]
        else:
            user_ip = self.request.META.get('REMOTE_ADDR')
        if not self.request.session.exists(self.request.session.session_key):
            self.request.session.create()
            print('!!!!!!!!!!New session')
            # The session has registered
            self.request.session[user_ip] = True

            host = self.request.session.session_key
            self.reset(request, host)

            data = self.search(host)
            #print('host is' + host)
            # return waiting sessions
            del self.request.session[user_ip]
            data['session_reply'] = -1
            return JsonResponse(data, status=status.HTTP_200_OK)
        else:
            # # 你先把这段uncomment掉后run twice search初始一下数据库,
            # # 然后再comment掉后跑
            # self.request.session.flush()
            # self.request.session.clear()
            # data = {
            #     "session_reply": 0
            # }
            # return JsonResponse(data, status=status.HTTP_200_OK)
            # # 你先把这段uncomment掉后run twice search初始一下数据库,
            # # 然后再comment掉后跑

            # Looks like the self.request.session.exists( get existed first
            host = self.request.session.session_key
            print('!!!!!!!!!!Session exist!!!, the host is' + host)
            # The session has registered
            print('The keys are', self.request.session.keys())
            if user_ip in self.request.session:
                print('xxxxxxxxxxxxxThe user has already made a request and is being processed')
                self.request.session.modified = True
                self.request.session.save()
                data = {
                    # "session_reply": "Your query is being processed, please wait and check back later (~5s)"
                    "session_reply": 0
                }
                return JsonResponse(data, status=status.HTTP_200_OK)
            elif len(self.request.session.keys()) != 0:
                print('xxxxxxxxxxxxxThe user hasn\'t made a request, and the GPUs are busy')
                data = {
                    # "session_reply": "GPUs are busy, please wait for other %d users and check back later" % (len(self.request.session.keys()))
                    "session_reply": len(self.request.session.keys())
                }
                return JsonResponse(data, status=status.HTTP_200_OK)
            else:
                print('xxxxxxxxxxxxxRequest accepted, The user can wait for their results')
                # self.request.session[user_ip] = [("in processing", {})]
                self.request.session[user_ip] = True
                print(self.request.session.keys())
                self.request.session.modified = True
                self.request.session.save()

                self.reset(request, host)
                # self.query = data.query
                data = self.search(host)
                # print(data)
                # print('host is' + host)
                # return waiting sessions
                # del self.request.session[user_ip]
                # self.request.session[user_ip] = [("in sending back", data)]
                del self.request.session[user_ip]
                print('!!!!!!!!!! SESSION HAS BEEN DELETED')
                self.request.session.modified = True
                self.request.session.save()
                data['session_reply'] = -1
                print('!!!!!!!!!! SESSION HAS BEEN DELETED 2')
                return JsonResponse(data, status=status.HTTP_200_OK)

    def reset(self, request, host):
        docset = Document.objects.filter(host=host)
        if docset.exists():
            print('|||||||||the type is', type(docset))
            Document.objects.filter(host=host).delete()
            Sentence.objects.filter(host=host).delete()
        print(request.data)
        self.query = request.data["query"]
    
    def extract_sentences(self, content):
        doc = nlp(content)
        return [sent.text for sent in doc.sents]

    def search(self, host):
        import math
        output, veracity = search_model(self.query)
        #Document.objects.all().delete()
        #Sentence.objects.all().delete()
        
        result_find = 0
        stance_list = ["Refute", "Neutral", "Support"]
        for document_i in output:
            document_info = output[document_i]
            #print(document_info)
            #print(math.isnan(document_info))
            #if math.isnan(document_info): 
            if type(document_info) != dict:
                print("pass", document_i)
                continue
            else:
                result_find += 1
            docscore = document_info["docscore"]
            #if math.exp(docscore*100) < 0.1:
            #    continue
            datatype = "Article"
            title = document_info["title"]
            content = document_info["content"]
            sent_list = self.extract_sentences(content)
            sent_similarity, sent_stance = match_stance(sent_list, document_info["sents"])
            sent_list = json.dumps(sent_list)
            sentiment = document_info["content_inference"]
            stance = stance_list[sentiment.index(max(sentiment))]
            url = document_info["url"]
            source = document_info["source"]
            document = Document(host=host, query=self.query,veracity=veracity["veracity_output"],veracity_confidence=veracity["veracity_outputs_probs"][0],sentence_list=sent_list,sent_stance_list=sent_stance,sent_similarity_list=sent_similarity,datatype=datatype,title=title,docscore=docscore,content=content,neg=sentiment[0],neu=sentiment[1],pos=sentiment[2],stance=stance,url=url,source=source)
            document.save()
            sentence_info = document_info["sents"]
            for sentence_i in sentence_info:
                sentence = sentence_info[sentence_i]
                sentence_text = sentence["sent"]
                content = sentence["sent_context"]
                sentiment = sentence["sent_inference"]
                similarity = sentence["sent_similarity"]
                stance = stance_list[sentiment.index(max(sentiment))]
                sentence = Sentence(host=host, sentence=sentence_text,content=content,neg=sentiment[0],neu=sentiment[1],pos=sentiment[2],stance=stance,similarity=similarity)
                sentence.save()
                document.sentence.add(sentence)
        if result_find == 0:
            #document = Document(query=self.query,veracity="Neutral",veracity_confidence=0,sentence_list=[],sent_stance_list=[],sent_similarity_list=0,datatype=datatype,title="No",docscore=0,content="",neg=0,neu=0,pos=0,stance="Negative",url="",source="")
            document = Document(host=host, query=self.query)
            document.save()
            return {}
        return output


class GetTwitter(APIView):
    def get(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        twitterset = Twitter.objects.all()
        serializer_class = TwitterSerializer(twitterset, many=True)
        return Response(serializer_class.data)

class GetVocab(APIView):
    def post(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        claim = Claim.objects.get(id=request.data["claim"])
        wordset = Vocabulary.objects.filter(claim=claim.claim, stance=request.data["stance"])
        serializer_class = VocabularySerializer(wordset, many=True)
        
        tweet_id = ["1221414", "5134131", "7245252"]
        query = "jim bakker's colloidal silver solution is a remedy for covid-19"
        tokenized_query = query.split(" ")
        top_tweet = bm25result.get_top_n(tokenized_query, tweet_id, n=2)
        #print("reload results",top_tweet)
        return Response(serializer_class.data)

class GetTweetSpread(APIView):
    def post(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        claim = Claim.objects.get(id=request.data["claim"])
        wordset = TweetSpread.objects.filter(tweet__claim=claim.claim)
        serializer_class = TweetSpreadSerializer(wordset, many=True)
        return Response(serializer_class.data)

class GetTweetMap(APIView):
    def post(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        claim = Claim.objects.get(id=request.data["claim"])
        #print(request.data["claim"])
        wordset = TweetMap.objects.filter(tweet__claim=claim.claim)
        #wordset = TweetMap.objects.all()
        serializer_class = TweetMapSerializer(wordset, many=True)
        return Response(serializer_class.data)

class GetTweetGraph(APIView):
    def post(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        claim = Claim.objects.get(id=request.data["claim"])
        wordset = TweetGraph.objects.filter(source__claim=claim.claim)
        claim_compare = Twitter.objects.values_list('claim', flat=True).distinct()[:5]
        for claim in claim_compare:
            wordset_compare = TweetGraph.objects.filter(source__claim=claim)
            wordset = wordset | wordset_compare
            #print(claim)
        serializer_class = TweetGraphSerializer(wordset, many=True)
        return Response(serializer_class.data)
    """def get(self, request, format=None):
        graphset = TweetGraph.objects.all()
        serializer_class = TweetGraphSerializer(graphset, many=True)
        return Response(serializer_class.data)"""

class GetTweetTopic(APIView):
    def post(self, request, format=None):
        #sentenceset = Sentence.objects.all()
        claim = Claim.objects.get(id=request.data["claim"])
        wordset = Topic.objects.filter(claim=claim.claim)
        #wordset = Topic.objects.all()
        serializer_class = TopicSerializer(wordset, many=True)
        return Response(serializer_class.data)

class GetClaim(APIView):
    def get(self, request, format=None):
        wordset = Claim.objects.all()
        serializer_class = ClaimSerializer(wordset, many=True)
        return Response(serializer_class.data)
    
class GetClaimInfo(APIView):
    def post(self, request, format=None):
        claim = Claim.objects.get(id=request.data["claim"])
        twitterset = Twitter.objects.filter(claim=claim.claim)
        #print(type(claim.claim), claim.claim, len(twitterset))
        #wordset = Claim.objects.all()
        #serializer_class = ClaimSerializer(wordset, many=True)
        return JsonResponse({"claim": claim.claim, "data_count": len(twitterset)})

class GetTweetCount(APIView):
    def post(self, request, format=None):
        claim = Claim.objects.get(id=request.data["claim"])
        wordset = TweetCount.objects.filter(claim=claim.claim)
        serializer_class = TweetCountSerializer(wordset, many=True)
        return Response(serializer_class.data)
        
class LoadLabelledData(APIView):
    
    def get(self, request, format=None, reset=True):
        if reset:
            Twitter.objects.all().delete()
            Claim.objects.all().delete()
            Vocabulary.objects.all().delete()
            TweetSpread.objects.all().delete()
            TweetMap.objects.all().delete()
            TweetGraph.objects.all().delete()
            TweetCount.objects.all().delete()
            Topic.objects.all().delete()
            TopicWords.objects.all().delete()

        """from debater_python_api.api.debater_api import DebaterApi
        debater_api = DebaterApi('90c57c807cfd41ad1eb996042e1039ceL05')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.sid = SentimentIntensityAnalyzer()
        self.pro_con_client = debater_api.get_pro_con_client()"""

        self.update_tweets()
        self.update_vocab()
        self.update_graph()
        self.update_tweetspread()
        self.update_tweetcount()
        self.update_topic()

        print("dataset updated")

        tweetset = Twitter.objects.all()[:10]
        serializer_class = TwitterSerializer(tweetset, many=True)
        #VocabularySerializer
        return Response(serializer_class.data)

    def update_tweets(self):
        data_path = os.path.join(project_dir, "data", "tweets")
        #label_dir = os.path.join(project_dir, "data", "COVID-RV.csv")
        label_dir = os.path.join(project_dir, "data", "Data_E", "COVID-RV.csv")
        
        self.info_dic = {}
        claims = []
        self.count_tweet_saved = 0
        with open(label_dir, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            tweet_save_dir = os.path.join(project_dir, "data", "Data_E", "conversations_withlabel")
            #claim[i], tweet[i], tweet_id[i], label[i], agreement[i]
            for row in spamreader:
                if self.count_tweet_saved >= 5000:
                    break
                comment_tweet = row[2] + ".jsonl"
                if os.path.isfile(os.path.join(tweet_save_dir, comment_tweet)):
                    tweet_source_path = os.path.join(tweet_save_dir, comment_tweet)
                else:
                    continue
                claim = row[0].lower()
                self.info_dic[row[2]] = [claim, row[3], row[1]]
                claims.append(claim)
                checkexist = Claim.objects.filter(claim=claim)
                if len(checkexist) == 0:
                    claim_to_save = Claim(claim=claim)
                    claim_to_save.save()
                    
                #comment
                # load tweet
                with open(tweet_source_path, "r") as f:
                    data = f.readlines()
                    f.close()
                for tweet in data:
                    tweet = json.loads(tweet)
                    checkexist = Twitter.objects.filter(tweet_id=tweet["id"])
                    if len(checkexist) == 0:
                        self.save_tweet(tweet, "comment", claim)
                            
                #source-tweet
                tweet_root_path = os.path.join(project_dir, "data", "Data_E", "conversations_root")
                if os.path.isfile(os.path.join(tweet_root_path, row[2]+".json")):
                    with open(tweet_source_path, "r") as f:
                        tweet = json.load(f)
                        f.close()
                else:
                    tweet = {"id": row[2], "text": row[1], "created_at": tweet["created_at"],  "public_metrics": {"retweet_count": len(data)}}
                self.save_tweet(tweet, "source", claim)
                
            print("update_tweets done")
            csvfile.close()
        self.claims = list(set(claims))


        """
        self.info_dic = {}
        claims = []
        with open(label_dir, newline='') as csvfile:  
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            for row in spamreader:
                #claim,tweet_id,label
                self.info_dic[row[1]] = [row[0].lower(), row[2]]
                claims.append(row[0].lower())
            csvfile.close()
        self.claims = list(set(claims))
        # save claims
        for claim in self.claims:
            checkexist = Claim.objects.filter(claim=claim)
            if len(checkexist) == 0:
                claim_to_save = Claim(claim=claim)
                claim_to_save.save()

        self.count_tweet_saved = 0
        
        #backup_path = os.path.join(project_dir, "data", "backup")
        for tweet_folder in os.listdir(data_path):
            #checkexist = Twitter.objects.filter(tweet_id=tweet_folder)
            #if len(checkexist) == 0:
            #source-tweet
            tweet_source_path = os.path.join(data_path, tweet_folder, "source-tweets", tweet_folder+".json")
            with open(tweet_source_path, "r") as f:
                tweet = json.load(f)
                tweet_id=tweet["id_str"]
                #claim = Claim.objects.filter(claim=self.info_dic[tweet_id][0])[0]
                self.save_tweet(tweet, "source", self.info_dic[tweet_id][0])
                f.close()

            #comment
            comment_path = os.path.join(data_path, tweet_folder, "comment_withlabel")
            for comment_tweet in os.listdir(comment_path):
                checkexist = Twitter.objects.filter(tweet_id=comment_tweet.split('.')[0])
                if len(checkexist) == 0:
                    tweet_source_path = os.path.join(comment_path, comment_tweet)
                    with open(tweet_source_path, "r") as f:
                        tweet = json.load(f)
                        self.save_tweet(tweet, "comment", self.info_dic[tweet_id][0])
                        f.close()

                
                #retweet
                retweet_path = os.path.join(data_path, tweet_folder, "retweet")
                for retweet_tweet in os.listdir(retweet_path):
                    checkexist = Twitter.objects.filter(tweet_id=retweet_tweet.split('.')[0])
                    if len(checkexist) == 0:
                        tweet_source_path = os.path.join(data_path, tweet_folder, "retweet", retweet_tweet)
                        with open(tweet_source_path, "r") as f:
                            tweet = json.load(f)
                            self.save_tweet(tweet, "retweet", claim)
                            f.close()
            """

            #move this tweet data folder to a backup dir
            #shutil.move(os.path.join(data_path, tweet_folder), os.path.join(backup_path, tweet_folder))
                
    def timeformatter(self, date):
        #"Sun Jan 26 18:01:42 +0000 2020"
        try:
            month_dic = {"Jan": "01", "Feb": "02", "Mar":"03", "Apr": "04", "May": "05", "Jun":"06", "Jul": "07", "Aug": "08", "Sep":"09", "Oct": "10", "Nov": "11", "Dec":"12"}
            split_date = date.split(" ")
            output_date = split_date[-1] + "-" + month_dic[split_date[1]] + "-" + split_date[2] + " " + split_date[3][:8]
            return output_date
        except: 
            return date[:10] + " " + date[11:19]
            
    
    def save_tweet(self, tweet, spread_type, claim):
        #check if this twitter exist in dataset (in case it appear in other tree as comment/source)
        checkexist = Twitter.objects.filter(tweet_id=tweet["id"])
        if len(checkexist) != 0:
            if spread_type == "comment":
                # in this case, this duplicate tweet must be source before, need to update
                tweet_comment = checkexist[0]
                tweet_comment.spread_type = spread_type
                tweet_comment.source_tweet_id = tweet['referenced_tweets']["id"]
                tweet_comment.save()
            return 0
        self.count_tweet_saved += 1
        if self.count_tweet_saved % 500 == 0:
            print("saved", self.count_tweet_saved)
        text = tweet['text']
        location, longitude, latitude = self.getlocation(tweet)
        
        #checkexist = ShowDate.objects.filter(date=time)
        #if len(checkexist) == 0:
        #    date = ShowDate(date=time)
        #    date.save()
        #else:
        #    date = checkexist.first()
        #a1.publications.add(p1)
        tweet_id = tweet["id"]
        retweet_count = tweet["public_metrics"]["retweet_count"]
        claim_id = Claim.objects.get(claim=claim).id
        #claim = Claim.objects.filter(claim=self.info_dic[tweet_id][0]).first()
        if spread_type == "source":
            time = self.timeformatter(tweet['created_at'])
            source_tweet_id = '0'
            stance = self.info_dic[tweet_id][1]
            if stance == "disagree":
                stance = 0
            elif stance == "agree":
                stance = 2
            else:
                stance = 1
            print(tweet_id, claim, stance)
        else:
            time = tweet['created_at'][:10] + " " + tweet['created_at'][11:19]
            #in_reply_to_status_id_str
            source_tweet_id = tweet['referenced_tweets'][0]['id']
            #stance = self.getstance(claim, text)
            stance = self.getstance(tweet)
            #print("source_tweet", tweet['in_reply_to_status_id_str'], tweet['source_tweet'])
        twitter = Twitter(text=text,location=location,longitude=longitude,latitude=latitude,time=time,tweet_id=tweet_id,source_tweet_id=source_tweet_id,spread_type=spread_type,retweet_count=retweet_count,comment_count=0,stance=stance,claim=claim,category=claim_id)
        twitter.save()
        #twitter.claim.add(claim)
        #twitter.show_date.add(date)
        if location != 'none':
            twitter_map = TweetMap(tweet=twitter, time=time)
            twitter_map.save()

    def getstance(self, tweet):
        score = (tweet["stance_score"] + tweet["sent_score"])/2
        #import random
        #score = random.uniform(-1, 1)
        #sentence_topic_dicts = [{'sentence' : text, 'topic' : claim.claim }]

        """sentence_topic_dicts = [{'sentence' : text, 'topic' : claim }]
        stance_score = self.pro_con_client.run(sentence_topic_dicts)[0]
        sent_score = self.sid.polarity_scores(text)['compound']
        score = (stance_score+sent_score)/2
        print(sentence_topic_dicts, self.stop_count, self.count_tweet_saved, stance_score, sent_score, score)
        """
        if score > 0.2:
            return 2
        elif score < 0:
            return 0
        else:
            return 1
        
    def getlocation(self, tweet):
        #input location, output longitude and latitude
        """if not isinstance(tweet['coordinates'], type(None)):
            return tweet['location'], tweet['coordinates'][0], tweet['coordinates'][1]
        if not isinstance(tweet['geo'], type(None)):
            return tweet['location'], tweet['geo']['coordinates'][1], tweet['geo']['coordinates'][0]
        if not isinstance(tweet['place'], type(None)):
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(tweet['place']['name']) + '?format=json'
            response = requests.get(url).json()
            if len(response) != 0:
                return tweet['user']['location'], response[0]['lon'], response[0]['lat']
            else:
                return 'none', 0, 0
        if tweet['user']['geo_enabled']:
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(tweet['user']['location']) + '?format=json'
            response = requests.get(url).json()
            #print("response", tweet['user']['location'], response)
            if len(response) != 0:
                return tweet['user']['location'], response[0]['lon'], response[0]['lat']
            else:
                return 'none', 0, 0"""
        #from random import randrange
        #location_choice = [["New York",-74.0060,40.7128], ["Washington DC",-77,15.38],["Los Angeles",-118,34], ["London",-0.1276,51.5072], ["Exeter",-3.5275,50.7260]]
        #return location_choice[randrange(5)]
        #return False, False
        if "location" in tweet:
            if tweet["location"] != "none":
                return tweet["location"], tweet["coordinates"][0], tweet["coordinates"][1]
        return 'none', 0, 0

    def update_vocab(self, num_topics = 10):
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer 
        import re
        
        lemmatizer = WordNetLemmatizer()
        #dateset = ShowDate.objects.all()
        claim_all = Claim.objects.values_list('claim', flat=True).distinct()
        print(claim_all)
        for claim in claim_all:
            tweetset = Twitter.objects.filter(claim=claim)
            word_set_support = {}
            word_set_refute = {}
            for tweet in tweetset:
                tweet_text = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.text).split()).rstrip()
                tweet_text = tweet_text.translate(str.maketrans('','',string.punctuation))
                tweet_text = tweet_text.lower()
                word_list = word_tokenize(tweet_text)
                word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words('english')]
                if len(word_list) < 2:
                    continue
                if tweet.stance == 2:
                    for i in word_list:
                        word_set_support[i] = word_set_support.get(i, 0) + 1
                elif tweet.stance == 0:
                    for i in word_list:
                        word_set_refute[i] = word_set_refute.get(i, 0) + 1
            chosen_words_support = sorted(word_set_support, key=word_set_support.get, reverse=True)[:100]
            chosen_words_refute = sorted(word_set_refute, key=word_set_refute.get, reverse=True)[:100]
            #print(claim, "chosen words",chosen_words_support[:10])
            for word in chosen_words_support:
                vocab = Vocabulary(word=word,count=word_set_support[word],claim=claim,stance=2)
                vocab.save()
            for word in chosen_words_refute:
                vocab = Vocabulary(word=word,count=word_set_refute[word],claim=claim,stance=0)
                vocab.save()
            print("done vocab updates")
            
    def update_topic(self):
        #get topics
        topic_path = os.path.join(project_dir, "data", "Data_E", "topic_info.json")
        with open(topic_path, "r") as f:
            topic_info = json.load(f)
            f.close()
        for claim in topic_info:
            topic_i = topic_info[claim]
            for i in range(len(topic_i["coordinates"])):
                x = round(float(topic_i["coordinates"][i][0])/10, 2)
                y = round(float(topic_i["coordinates"][i][1])/10, 2)
                print(x, y)
                #x, y, top_sentence = self.get_topic_info(top_words, embedding_dic, sentence_embedding_dic)
                topic = Topic(claim=claim, x=x, y=y, weight=float(topic_i["weight"][i]),text=topic_i["top_sentence"][i])
                topic.save()
                for word in topic_i["top_words"][i]:
                    word = TopicWords(word=word[0], weight=float(word[1]))
                    word.save()
                    topic.topicwords.add(word)
            print("topic words saved")
        """
            lda_model = self.topicmodel(list(sentence_embedding_dic.values()), num_topics)
            glove_path = os.path.join(project_dir, "data", "glove.6B.200d.txt")
            embedding_dic = self.load_glove(glove_path)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            print("start build sentence_embedding_dic", len(list(sentence_embedding_dic.keys())))
            for tweet_text in sentence_embedding_dic:
                sentence_embedding_dic[tweet_text] = self.get_sentence_embeddings(sentence_embedding_dic[tweet_text], embedding_dic)
                #tweet_embeddings = [self.get_sentence_embeddings(word_list, embedding_dic) for word_list in doc_list]
            print("start topic modelling")
            topic_embeddings = []
            topic_sentences = []
            for i in range(num_topics):
                top_words = lda_model.show_topic(i)
                reweight = 0
                for word in top_words:
                    if word[0] in embedding_dic:
                        reweight += word[1]
                        try:
                            topic_embedding += word[1]*embedding_dic[word[0]]
                        except:
                            topic_embedding = word[1]*embedding_dic[word[0]]
                    else:
                        print(word, "not in glove")
                topic_embeddings.append(topic_embedding/reweight)
                topic_sentences.append(self.find_closest_embeddings(sentence_embedding_dic, topic_embedding, "euclidean")[0])
            print("start PCA for visualisation")
            pca.fit(topic_embeddings)
            #x, y = pca.singular_values_
            for i in range(num_topics):
                top_words = lda_model.show_topic(i)
                #x, y, top_sentence = self.get_topic_info(top_words, embedding_dic, sentence_embedding_dic)
                topic = Topic(x=pca.singular_values_[i][0], y=pca.singular_values_[i][1], text=topic_sentences[i])
                for word in top_words:
                    word = TopicWords(topic=topic, word=word[0], weight=word[1])
            print("topic words saved")"""
        #.update(field1='some value')



                
    def update_graph(self):
        date_all = list(Twitter.objects.values_list('time', flat=True).distinct())
        date_all.sort()
        for date in date_all:
            #tweetidset = Twitter.objects.filter(time=date).values_list('source_tweet_id', flat=True).distinct()
            sourcetweets = Twitter.objects.filter(time=date)
            for sourcetweet in sourcetweets:
                targettweets = Twitter.objects.filter(source_tweet_id=sourcetweet.tweet_id)
                if len(targettweets) > 0: 
                    for targettweet in targettweets:
                        tweetgraph = TweetGraph(source=sourcetweet,target=targettweet,time=date)
                        tweetgraph.save()
        print("graph updated")

    def update_tweetspread(self):
        tweetset = Twitter.objects.filter(spread_type='source')
        for tweet in tweetset:
            comment_count = len(Twitter.objects.filter(source_tweet_id=tweet.tweet_id))
            direct_spread = tweet.retweet_count + comment_count
            total_spread, stance_count = self.count_totalspread(tweet, count=0, stance_count=[0,0,0])
            tweetspread = TweetSpread(tweet=tweet,direct_spread=direct_spread,total_spread=total_spread,support=stance_count[2],neutral=stance_count[1],refute=stance_count[0])
            tweetspread.save()
        tweetset = Twitter.objects.all()
        for tweet in tweetset:
            comment_count = len(Twitter.objects.filter(source_tweet_id=tweet.tweet_id))
            #tweet = Twitter.objects.get(tweet_id=tweet.tweet_id)
            if comment_count != 0:
                tweet.comment_count = comment_count
                tweet.save()

        print("tweetspread updated")

    def count_totalspread(self, tweet, count=0, stance_count=[0,0,0]):
        #support=0, neutral=0, refute=0
        tweetset = Twitter.objects.filter(source_tweet_id=tweet.tweet_id)
        count = count + 1 + tweet.retweet_count
        stance_count[tweet.stance] = stance_count[tweet.stance] + 1 + tweet.retweet_count
        for tweet_i in tweetset:
            count, stance_count = self.count_totalspread(tweet_i, count, stance_count)
        return count, stance_count
        
    def update_tweetcount(self):
        for claim in self.claims:
            date_all = Twitter.objects.values_list('time', flat=True).distinct()
            for date in date_all:
                count = len(Twitter.objects.filter(claim=claim, time=date))
                tweetspread = TweetCount(claim=claim,count=count,time=date)
                tweetspread.save()
        print("tweetcount updated")


class LoadUnlabelledData(APIView):
    def get(self, request, format=None):
        pass

class InputTweetQuery(APIView):
    def post(self, request, format=None):
        self.query = request.data["query"]
        self.date_select = request.data["date"]

        checkexist = ShowDate.objects.filter(claim=self.query, date=self.date_select)
        if len(checkexist) == 0:
            date = ShowDate(date=self.date_select)
            date.save()
            self.beamsearch()
        else:
            serializer_class = TwitterSerializer(checkexist, many=True)
            return Response(serializer_class.data)

        
    def beamsearch(self):
        pass
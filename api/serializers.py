from rest_framework import serializers
from .models import Claim, Room, Document, Sentence, TweetGraph, TweetMap, TweetSpread, Vocabulary, Twitter, TweetCount, Topic, TopicWords

# Empty database
from django.contrib.sessions.models import Session


class RoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fields = ('id', 'code', 'host', 'guest_can_pause',
                  'votes_to_skip', 'created_at')


class CreateRoomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Room
        fields = ('guest_can_pause', 'votes_to_skip')


class SentenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentence
        fields = ('id', 'sentence', 'content',
                  'neg', 'neu', 'pos', 'stance', 'similarity')


# return document info for document list page
class DocumentRowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ('id', 'query', 'veracity', 'veracity_confidence', 'datatype', 'title', 'docscore', 'content', 'neg', 'neu', 'pos', 'stance', 'url', 'source')
 
 
# return sentence info for sentence list page
class DocInfor(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ('id', 'query', 'veracity', 'veracity_confidence', 'sentence_list', 
                    'sent_stance_list', 'sent_similarity_list', 'datatype', 'title',
                    'content', 'neg', 'neu', 'pos', 'url', 'source')
class SentenceRowSerializer(serializers.ModelSerializer):
    documents = DocInfor(many=True)
    class Meta:
        model = Sentence
        fields = ('id', 'sentence', 'content',
                  'neg', 'neu', 'pos', 'stance', 'similarity', 'documents')


class TopicWordsSerializer(serializers.ModelSerializer):
    class Meta:
        model = TopicWords
        fields = ('word', 'weight')
class TopicSerializer(serializers.ModelSerializer):
    topicwords = TopicWordsSerializer(many=True)
    class Meta:
        model = Topic
        fields = ('id', 'topicwords', 'claim', 'x', 'y', 'weight', 'text')
        
#return detailed information of document
class DocumentDetailsSerializer(serializers.ModelSerializer):
    sentence = SentenceSerializer(many=True)
    class Meta:
        model = Document
        fields = ('id', 'sentence', 'query', 'veracity', 'veracity_confidence', 'sentence_list', 
                    'sent_stance_list', 'sent_similarity_list', 'datatype', 'title', 'docscore',
                    'content', 'neg', 'neu', 'pos', 'url', 'source')

    def create(self, validated_data):
        sentences = validated_data.pop('sentence')# if 'sentences' in validated_data else []
        document = Document.objects.create(**validated_data)
        for sentence in sentences:
            Sentence.object.create(document=document, **sentence)
        #document.sentences.set(sentences)
        return document

# return sentence info for sentence list page
# ShowDate, Vocabulary, Twitter
class VocabularySerializer(serializers.ModelSerializer):
    class Meta:
        model = Vocabulary
        fields = ('word', 'count')

class ClaimSerializer(serializers.ModelSerializer):
    class Meta:
        model = Claim
        fields = ('id', 'claim')

class TwitterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Twitter
        fields = ('text', 'time', 'tweet_id', 'source_tweet_id', 'stance', 'claim')

class TweetCountSerializer(serializers.ModelSerializer):
    class Meta:
        model = TweetCount
        fields = ('count', 'time')

class TweetforSpreadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Twitter
        fields = ('text', 'time', 'tweet_id', 'source_tweet_id', 'stance')
class TweetSpreadSerializer(serializers.ModelSerializer):
    tweet = TweetforSpreadSerializer(read_only=True)
    class Meta:
        model = TweetSpread
        fields = ('id', 'tweet', 'direct_spread', 'total_spread', 'support', 'neutral', 'refute')

# return representative tweet info for TweetSpread plot
class TweetforMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = Twitter
        fields = ('text', 'location', 'longitude', 'latitude', 'stance')
class TweetMapSerializer(serializers.ModelSerializer):
    tweet = TweetforMapSerializer(read_only=True)
    class Meta:
        model = TweetMap
        fields = ('id', 'tweet', 'time')

# return tweet link info for TweetGraph plot
class TweetforGraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = Twitter
        fields = ('id', 'text', 'time', 'spread_type', 'stance', 'claim', 'retweet_count', 'comment_count')
class TweetGraphSerializer(serializers.ModelSerializer):
    source = TweetforGraphSerializer(read_only=True)
    target = TweetforGraphSerializer(read_only=True)
    class Meta:
        model = TweetGraph
        fields = ('source', 'target')

#def __init__(self, *args, **kwargs):
    # super(MySerializer, self).__init__(*args, **kwargs)
    # self.random_id = randint(1, 5)
    #Session.objects.all().delete()

    """        # the claim that this tweet related to
    claim = models.CharField(max_length=500)

    # content of twitter
    text = models.CharField(max_length=500)

    # city name of the location
    location = models.CharField(max_length=50)

    # longitude (x) and latitude(y) of the location
    longitude = models.DecimalField(max_digits=5, decimal_places=2)
    latitude = models.DecimalField(max_digits=5, decimal_places=2)
    
    # post time, in format 2021-07-29
    time = models.CharField(max_length=50)

    # tweet_id
    tweet_id = models.CharField(max_length=50) 

    # tweet_id of source tweet, set to be 0 if no source tweet
    source_tweet_id = models.CharField(max_length=50) 

    # commend or retweet
    spread_type = models.CharField(max_length=50) 

    # date need to show this tweet
    show_date = models.ForeignKey(ShowDate)"""

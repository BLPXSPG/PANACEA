from django.db import models
import string
import random


def generate_unique_code():
    length = 6

    while True:
        code = ''.join(random.choices(string.ascii_uppercase, k=length))
        if Room.objects.filter(code=code).count() == 0:
            break

    return code


class Room(models.Model):
    code = models.CharField(
        max_length=8, default=generate_unique_code, unique=True)
    host = models.CharField(max_length=50, unique=True)
    guest_can_pause = models.BooleanField(null=False, default=False)
    votes_to_skip = models.IntegerField(null=False, default=1)
    created_at = models.DateTimeField(auto_now_add=True)


#class ShowDate(models.Model):
    # user selected data, showing tweet that post, comment, retweet on this day and their source tweet.
    #date = models.CharField(max_length=50)


class Claim(models.Model):
    # the claim that user input
    claim = models.CharField(max_length=500)


class Vocabulary(models.Model):
    # word
    word = models.CharField(max_length=50)

    # word count
    count = models.PositiveIntegerField()

    # post time, in format 2021-07-29
    #show_date = models.ForeignKey(ShowDate, on_delete=models.CASCADE)
    #time = models.CharField(max_length=50)

    # claim
    claim = models.CharField(max_length=300)

    # stance, 0 refute, 1 neutral, 2 support
    stance = models.PositiveIntegerField()

    class Meta: 
        unique_together = ('word', 'claim', 'stance',)


class TopicWords(models.Model):
    # corresponding topic
    #topic = models.ForeignKey(Topic, on_delete=models.CASCADE)
    
    # word
    word = models.CharField(max_length=50)
    
    # word weight
    weight = models.DecimalField(max_digits=10, decimal_places=8)
    
    
class Topic(models.Model):
    # claim
    claim = models.CharField(max_length=300)
    
    # position
    x = models.DecimalField(max_digits=5, decimal_places=2)
    y = models.DecimalField(max_digits=5, decimal_places=2)
    
    #topic words
    topicwords = models.ManyToManyField(TopicWords, related_name='topicwords')
    
    # topic weight
    weight = models.DecimalField(max_digits=10, decimal_places=8)
    
    #topic sentence
    text = models.CharField(max_length=500)
    
    
class Twitter(models.Model):

    # content of twitter
    text = models.CharField(max_length=500)

    # city name of the location
    location = models.CharField(max_length=50)

    # longitude (x) and latitude(y) of the location
    longitude = models.DecimalField(max_digits=5, decimal_places=2)
    latitude = models.DecimalField(max_digits=5, decimal_places=2)
    
    # post time, in format 2021-07-29 22:21:23
    time = models.CharField(max_length=50)

    # tweet_id
    tweet_id = models.CharField(max_length=50) 

    # tweet_id of source tweet, set to be 0 if no source tweet
    source_tweet_id = models.CharField(max_length=50) 

    # comment or retweet
    spread_type = models.CharField(max_length=50) 

    # number of retweet
    retweet_count = models.PositiveIntegerField()

    #number of comment
    comment_count = models.PositiveIntegerField()

    # date need to show this tweet
    #show_date = models.ManyToManyField(ShowDate, related_name='show_date')

    # the claim that this tweet related to
    #claim = models.ManyToManyField(Claim, related_name='claims', null=True)
    claim = models.CharField(max_length=300)

    #category, the id of claim
    category = models.PositiveIntegerField()

    # stance, 0 refute, 1 neutral, 2 support
    stance = models.PositiveIntegerField()


class TweetSpread(models.Model):
    # tweet (text, time)
    tweet = models.ForeignKey(Twitter, on_delete=models.CASCADE)

    # number of direct retweet and comment of this tweet (not includes it self)
    direct_spread = models.PositiveIntegerField()

    # total number of retweet of the tweet and its children (includes it self)
    total_spread = models.PositiveIntegerField()

    # number of support
    support = models.PositiveIntegerField()

    # number of neutral
    neutral = models.PositiveIntegerField()

    # number of refute
    refute = models.PositiveIntegerField()


class TweetMap(models.Model):
    # tweet (text, location, longitude (x),  latitude(y), spread type, claim, 
    tweet = models.ForeignKey(Twitter, on_delete=models.CASCADE)

    # date of the tweet
    time = models.CharField(max_length=50)


class TweetGraph(models.Model):
    # source tweet
    source = models.ForeignKey(Twitter, on_delete=models.CASCADE, related_name='source')

    # target tweet
    target = models.ForeignKey(Twitter, on_delete=models.CASCADE, related_name='target')

    # date of the source tweet (x)
    time = models.CharField(max_length=50)

    # the position let points on the same date equally spread - leave to the frontend
    #y = models.DecimalField(max_digits=5, decimal_places=2)
    class Meta: 
        unique_together = ('source', 'target')


class TweetCount(models.Model):
    # claim
    claim = models.CharField(max_length=300)

    # count of related tweets about the claim on this date 
    count = models.PositiveIntegerField()

    # date of the source tweet
    time = models.CharField(max_length=50)


class Sentence(models.Model):

    #who search this query
    host = models.CharField(max_length=50)

    # sentence text, CharField(text)
    sentence = models.CharField(max_length=200)

    #sentence context, CharField(text)
    content = models.CharField(max_length=500)

    #sentence inference
    neg = models.DecimalField(max_digits=5, decimal_places=2)
    neu = models.DecimalField(max_digits=5, decimal_places=2)
    pos = models.DecimalField(max_digits=5, decimal_places=2)

    #stance
    stance = models.CharField(max_length=10, null=True)
    
    #sentence similarity to query
    similarity = models.DecimalField(max_digits=5, decimal_places=2)


# Create your models here.
class Document(models.Model):

    #who search this query
    host = models.CharField(max_length=50)

    # corresponding sentences
    sentence = models.ManyToManyField(Sentence, related_name='documents')

    # input query
    query = models.CharField(max_length=300, null=True)

    # veracity of query
    veracity = models.BooleanField(null=True)

    #confidence of veracity
    veracity_confidence = models.DecimalField(max_digits=5, decimal_places=2, null=True)

    #sentence list
    sentence_list = models.CharField(max_length=5000, null=True)
    
    #sentence stance list
    sent_stance_list = models.CharField(max_length=300, null=True)

    #sentence similarity list
    sent_similarity_list = models.CharField(max_length=300, null=True)
    
    #more data dype: https://docs.djangoproject.com/en/2.0/ref/models/fields/#django.db.models.IntegerField
    # document id, IntegerField(integer)
    #query = models.IntegerField(default=0)

    #document type, can be article, twitter, etc.
    datatype = models.CharField(max_length=10, null=True)

    #title
    title = models.CharField(max_length=100, null=True)
    
    #document abstract
    #abstract = models.CharField(max_length=200, null=True)

    #document confidence score
    docscore = models.DecimalField(max_digits=5, decimal_places=2, null=True)

    # dpcument content, CharField(text)
    content = models.CharField(max_length=1000, null=True)

    #document inference
    neg = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    neu = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    pos = models.DecimalField(max_digits=5, decimal_places=2, null=True)

    #stance
    stance = models.CharField(max_length=10, null=True)

    # document link: must be the format without https://
    url = models.CharField(max_length=100, null=True)

    # document source
    source = models.CharField(max_length=20, null=True)

    # sentences

    # top 3 sentences





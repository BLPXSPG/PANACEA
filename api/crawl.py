import tweepy
import os
import csv
import json

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

def get_retweets(tweet_id, n_retweet):
	retweet_list = []
	for retweet in api.get_retweets(tweet_id, count=n_retweet):
		retweet_list.append(retweet)
		print("retweet", n_retweet, retweet.id, retweet.retweet_count)
		newTweets = api.search_tweets(q="to:"+str(retweet.user.id), count=5, tweet_mode="extended", since_id=retweet.id)
		for tweet in newTweets:
			print("-----", tweet.full_text.encode('utf-8'), tweet.id)
	return retweet_list

def get_tweetcontent(source_tweet_id, id_structure, save_dir):
	try:
		source_tweet = api.get_status(source_tweet_id, tweet_mode="extended")._json 
		#retweet_list = get_retweets(source_tweet_id, source_tweet['retweet_count'])
		print("collected", source_tweet_id, source_tweet['full_text'], source_tweet['retweet_count'])
		print()
	except Exception as e: 
		# if source tweet is deleted, then we discard the whole spread tree
		print(e, "source tweet deleted", source_tweet_id)
		return False

	id_list = get_id_list(source_tweet_id, id_structure[source_tweet_id], id_list = [])
	tweet_get = []
	print("tweet count under id",source_tweet_id, len(id_list))
	for tweet_id in id_list:
		try:
			tweet_info = api.get_status(tweet_id[1], tweet_mode="extended")._json
			tweet_info["source_tweet"] = tweet_id[0]
			tweet_get.append([tweet_id, tweet_info])
			#print("collected", tweet_id, tweet_info["text"], tweet_info["source_tweet"])
			#print("comment", tweet_id, tweet_info['full_text'], tweet_info['retweet_count'])
		except:
			pass
		#print(tweet_info)
	if len(tweet_get) < 1:
		print(e, len(tweet_get))
		return False
	source_path = os.path.join(save_dir, "source-tweets")
	comment_path = os.path.join(save_dir, "comment")
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
		os.makedirs(source_path)
		os.makedirs(comment_path)
	with open(os.path.join(source_path, str(source_tweet_id)+".json"), "w") as f:
		json.dump(source_tweet, f)
		f.close()
	for comment_tweet in tweet_get:
		with open(os.path.join(comment_path, str(comment_tweet[0][1])+".json"), "w") as f:
			json.dump(comment_tweet[1], f)
			f.close()
	
def get_id_list(source_tweet, id_dic, id_list = []):
	if id_dic != {}:
		for key in id_dic:
			id_list += [[source_tweet, key]]
			id_list = get_id_list(key, id_dic[key], id_list)
	return id_list

project_dir = os.path.abspath(os.path.dirname(__file__))
id_path = os.path.join(project_dir, "data", "COVID-RV.csv")
claim_dic = {}
data_dic = {}
claim_count = 0
with open(id_path, "r") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	head = next(csv_reader)
	line_count = 0
	for row in csv_reader:
		line_count += 1
		#claim,tweet_id,label
		claim = row[0]
		tweet_id = row[1]
		label = row[2]
		conversation_path = os.path.join(project_dir, "data", "conversation_structures", "structure_"+str(tweet_id)+".json")
		with open(conversation_path, "r") as f:
			id_structure = json.load(f)
			f.close()
		save_dir = os.path.join(project_dir, "data", "tweets", str(tweet_id))
		get_tweetcontent(tweet_id, id_structure, save_dir)
		
		#tweet_info = api.get_status(tweet_id, tweet_mode="extended") 


		if claim not in claim_dic:
			claim_dic[claim] = claim_count
			claim_count += 1
	
		
	
"""
claims = ['covid', 'vaccine']
for claim in claims:
    print(claim)
    newTweets = api.search_tweets(q=claim, count=tweetsPerQry, tweet_mode="extended")
    for tweet in newTweets:
        print(tweet.full_text.encode('utf-8'))
        print(tweet.id)
        print(retweets(tweet.id, 3))
    print()"""
    
"""maxId = -1
tweetCount = 0
while tweetCount < maxTweets:
	if(maxId <= 0):
		newTweets = api.search_tweets(q=hashtag, count=tweetsPerQry, tweet_mode="extended")
	else:
		newTweets = api.search_tweets(q=hashtag, count=tweetsPerQry, max_id=str(maxId - 1), tweet_mode="extended")

	if not newTweets:
		print("Tweet Habis", newTweets)
		break
	
	for tweet in newTweets:
		print(tweet.full_text.encode('utf-8'))
		
	tweetCount += len(newTweets)	
	maxId = newTweets[-1].id"""





	
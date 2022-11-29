from rank_bm25 import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]
tweet_id = ["1221414", "5134131", "7245252"]

tokenized_corpus = [doc.split(" ") for doc in corpus]


bm25 = BM25Okapi(tokenized_corpus)
query = "windy London"
tokenized_query = query.split(" ")
top_tweet = bm25.get_top_n(tokenized_query, corpus, n=2)
print("retrieve results", top_tweet)


import pickle

#To save bm25 object
with open('bm25result', 'wb') as bm25result_file:
    pickle.dump(bm25, bm25result_file)
    
with open('bm25result', 'rb') as bm25result_file:
    bm25result = pickle.load(bm25result_file)
    
top_tweet = bm25result.get_top_n(tokenized_query, tweet_id, n=2)
print("reload results",top_tweet)


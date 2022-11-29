from django.urls import path
from .views import InputQuery, GetDocument, GetSentence, GetDocumentDetails, checkQueueState
from .views import InputTweetQuery, GetVocab, GetTwitter, GetClaim, GetTweetCount, GetTweetSpread, GetTweetMap, GetTweetGraph, LoadLabelledData, LoadUnlabelledData, GetClaimInfo, GetTweetTopic

urlpatterns = [
    path('input-query', InputQuery.as_view()),
    path('get-document', GetDocument.as_view()),
    path('get-sentence', GetSentence.as_view()),
    path('get-document-details', GetDocumentDetails.as_view()),
    path('check-queue-state', checkQueueState.as_view()),
    path('input-tweet-query', InputTweetQuery.as_view()),
    path('get-vocabulary', GetVocab.as_view()),
    path('get-tweet', GetTwitter.as_view()),
    path('get-tweetcount', GetTweetCount.as_view()),
    path('get-claim', GetClaim.as_view()),
    path('get-claiminfo', GetClaimInfo.as_view()),
    path('get-tweetspread', GetTweetSpread.as_view()),
    path('get-tweetmap', GetTweetMap.as_view()),
    path('get-tweetgraph', GetTweetGraph.as_view()),
    path('get-tweettopic', GetTweetTopic.as_view()),
    path('load-labelled-data', LoadLabelledData.as_view()),
    path('load-unlabelled-data', LoadUnlabelledData.as_view()),
]

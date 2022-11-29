from django.urls import path
from .views import index

urlpatterns = [
    path('', index),
    path('panacea', index),
    path('search', index),
    path('trace/<str:id>', index),
    path('queue', index),
    path('documents', index),
    path('sentences', index),
    path('document/<str:id>', index),
]
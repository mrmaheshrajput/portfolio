from django.urls import path

from news.views import NewsHomeView

app_name    = 'news'
urlpatterns = [
    path('',NewsHomeView.as_view(), name='news_home_view'),
]

from django.urls import path

from kdd.views import KddHomeView, PredictionsView

app_name    = 'kdd'
urlpatterns = [
    path('',KddHomeView.as_view(), name='kdd_home_view'),
    path('predict/',PredictionsView.as_view(), name='predictions_view'),
]

from django.urls import path

from kdd.views import DashboardView, PredictionsView

app_name    = 'kdd'
urlpatterns = [
    path('',DashboardView.as_view(), name='dashboard'),
    path('predict/',PredictionsView.as_view(), name='predictions_view'),
]

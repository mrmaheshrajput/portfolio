from django.urls import path

from kdd.views import DashboardView

app_name    = 'kdd'
urlpatterns = [
    path('',DashboardView.as_view(), name='dashboard')
]

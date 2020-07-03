import random
import json
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.generic import View
from django.urls import reverse

from .models.data import data_dict
from .models.main import ScoreModel, CustomScaler

class DashboardView(View):
    template_name               = 'kdd/dashboard_view.html'

    def get(self, request):
        return render(request, self.template_name)

class PredictionsView(View):

    def get(self, request):

        object = data_dict[random.randint(1,11)]

        pred = ScoreModel('churn','appetency','upselling','scaler','freq_encodings')
        pred.load_and_clean_data(object)
        response = pred.predict()
        # print(response)
        return HttpResponse(
            json.dumps(response),
            content_type="application/json"
        )

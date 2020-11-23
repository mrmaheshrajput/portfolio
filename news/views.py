import re
import json
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.generic import View
from django.urls import reverse

from .models.main import NewsPredictor

class NewsHomeView(View):
    template_name               = 'news/news_home_view.html'
    # object                      = NewsPredictor('vectorizer','model')

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        query                   = request.POST['news_input']

        if query:
            object              = NewsPredictor('vectorizer','model')
            object.load_and_clean_data(query)
            output              = object.predict()
            print(output)
            return render(request, self.template_name, {'output':output,'query':query})

        else:
            return render(request, self.template_name, {'query':'You are supposed to write something here...'})

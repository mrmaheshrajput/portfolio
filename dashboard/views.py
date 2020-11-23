import random
import json
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.generic import View
from django.urls import reverse


class DashboardView(View):
    template_name               = 'dashboard/dashboard_view.html'

    def get(self, request):
        return render(request, self.template_name)

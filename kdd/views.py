from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.generic import View
from django.urls import reverse


class DashboardView(View):

    def get(self, request):
        return HttpResponse('Test !')

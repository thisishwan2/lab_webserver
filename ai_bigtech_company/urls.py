from django.urls import path

from ai_bigtech_company import views

app_name = 'ai_bigtech_company'

urlpatterns = [
    path("", views.index, name='index'),
    ]
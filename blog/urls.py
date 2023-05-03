from django.urls import path
from blog.views import *
from blog.models import *

urlpatterns = [
    path('', Blog.as_view(), name='blog'),
]
from django.urls import path

from home import views
from home.views import *

urlpatterns =[

    path('', Index.as_view(), name='home'),
    path('recommend/', DisplayRecommendation.as_view(), name='display-rec'),
    path('upload-review/', UploadReview.as_view(), name='review'),
    path('wishlist/', Wish.as_view(), name='wish'),
    path('about/', About.as_view(), name='about'),
    path('history/', History.as_view(), name='history'),


    path('dev/test/', Test.as_view(), name='test'),
    path('dev/image/', ImageUpload.as_view(), name='img-upload'),
    path('dev/gg/', XYZ.as_view(), name='xyz'),
    path('dev/metrics/', Metrics.as_view(), name='metrics'),
]
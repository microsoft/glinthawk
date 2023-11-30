from django.contrib import admin
from django.urls import path, include

from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns

import prompts.views

router = routers.DefaultRouter()
router.register(r'jobs', prompts.views.JobViewSet)
router.register(r'prompts', prompts.views.PromptViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace='rest_framework')),
]

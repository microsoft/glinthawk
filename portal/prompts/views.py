from django.shortcuts import render

from rest_framework import viewsets
from rest_framework import permissions

from .models import Prompt
from .serializers import PromptSerializer

class PromptViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows prompts to be viewed or edited.
    """
    queryset = Prompt.objects.all().order_by('-submitted_at')
    serializer_class = PromptSerializer
    permission_classes = [permissions.IsAuthenticated]

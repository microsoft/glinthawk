from django.contrib.auth.models import User
from rest_framework import serializers

from .models import Prompt

class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["hash", "status", "submitted_at", "preprocessed_at", "completed_at"]
        read_only_fields = ["hash", "submitted_at"]

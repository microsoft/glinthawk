from django.contrib.auth.models import User
from rest_framework import serializers, exceptions

import base58

from .models import Job, Prompt

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = ["language_model", "file", "created_at", "completed_at"]

    def create(self, validated_data):
        validated_data["created_by"] = self.context["request"].user
        return super().create(validated_data)

class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["job", "hash"]

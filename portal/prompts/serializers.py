from django.contrib.auth.models import User
from rest_framework import serializers

import base58

from .models import Prompt

class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["hash", "status", "submitted_at", "preprocessed_at", "completed_at"]
        read_only_fields = ["hash", "submitted_at"]

class PromptSubmissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = ["text", "language_model"]
        read_only_fields = ["hash", "submitted_at"]

    def create(self, validated_data):
        prompt_text = validated_data["text"]
        return Prompt.objects.create(
            user=self.context["request"].user,
            hash=base58.b58encode(prompt_text).decode(),
            container="test",
            origin="test",
            language_model=validated_data["language_model"],
            status=1,
        )

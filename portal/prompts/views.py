import base58

from django.shortcuts import render
from rest_framework import viewsets, permissions, generics

from .models import Prompt
from .serializers import PromptSerializer


class PromptViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows prompts to be viewed or edited.
    """

    queryset = Prompt.objects.all().order_by("submitted_at")
    serializer_class = PromptSerializer
    permission_classes = [permissions.IsAuthenticated]


class SubmitPromptView(generics.CreateAPIView):
    """
    API endpoint that allows prompts to be submitted.
    """

    queryset = Prompt.objects.all().order_by("submitted_at")
    serializer_class = PromptSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        prompt_text = serializer.validated_data["text"]
        serializer.save(
            user=self.request.user,
            hash=base58.b58encode(prompt_text).decode(),
            container="test",
            origin="test",
            language_model="llama2_7b",
            status=1,
        )

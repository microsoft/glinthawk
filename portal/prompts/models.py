from django.db import models
from django.contrib.auth.models import User


class Job(models.Model):
    class LanguageModel(models.TextChoices):
        LLAMA2_7B = "llama2_7b", "LLaMa2 7B"
        LLAMA2_7B_CHAT = "llama2_7b_chat", "LLaMa2 7B Chat"
        LLAMA2_70B = "llama2_70b", "LLaMa2 70B"
        LLAMA2_70B_CHAT = "llama2_70b_chat", "LLaMa2 70B Chat"

    language_model = models.CharField(
        max_length=32, choices=LanguageModel.choices, default=LanguageModel.LLAMA2_7B
    )

    file = models.FileField(upload_to="raw/", blank=True, null=True)
    url = models.URLField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)


class Prompt(models.Model):
    class Status(models.IntegerChoices):
        SUBMITTED = 1, "Submitted"
        PREPROCESSING = 2, "Preprocessing"
        PREPROCESSED = 3, "Preprocessed"
        COMPLETED = 4, "Completed"

    job = models.ForeignKey(Job, on_delete=models.CASCADE, null=True)
    hash = models.CharField(max_length=64)
    status = models.IntegerField(choices=Status.choices, default=Status.SUBMITTED)
    preprocessed_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

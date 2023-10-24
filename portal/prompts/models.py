from django.db import models
from django.contrib.auth.models import User

class Prompt(models.Model):
    class Status(models.IntegerChoices):
        SUBMITTED = 1, "Submitted"
        PREPROCESSING = 2, "Preprocessing"
        PREPROCESSED = 3, "Preprocessed"
        COMPLETED = 4, "Completed"

    class LanguageModel(models.TextChoices):
        LLAMA2_7B = "llama2_7b", "LLaMa2 7B"
        LLAMA2_7B_CHAT = "llama2_7b_chat", "LLaMa2 7B Chat"
        LLAMA2_70B = "llama2_70b", "LLaMa2 70B"
        LLAMA2_70B_CHAT = "llama2_70b_chat", "LLaMa2 70B Chat"


    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    hash = models.CharField(max_length=64)
    container = models.CharField(max_length=256)
    origin = models.CharField(max_length=256, null=True)
    language_model = models.CharField(max_length=32, choices=LanguageModel.choices)
    status = models.IntegerField(choices=Status.choices, default=Status.SUBMITTED)
    submitted_at = models.DateTimeField(auto_now_add=True)
    preprocessed_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)

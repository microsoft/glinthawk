import uuid

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


def job_directory_path(instance, filename):
    return "raw/user_{0}/{1}".format(instance.created_by.id, str(instance.uuid) + ".zip")


class Job(models.Model):
    class LanguageModel(models.TextChoices):
        LLAMA2_7B = "llama2_7b", "LLaMa2 7B"
        LLAMA2_7B_CHAT = "llama2_7b_chat", "LLaMa2 7B Chat"
        LLAMA2_70B = "llama2_70b", "LLaMa2 70B"
        LLAMA2_70B_CHAT = "llama2_70b_chat", "LLaMa2 70B Chat"

    uuid = models.UUIDField(
        unique=True, editable=False, primary_key=True, default=uuid.uuid4
    )

    language_model = models.CharField(
        max_length=32, choices=LanguageModel.choices, default=LanguageModel.LLAMA2_7B
    )

    file = models.FileField(
        upload_to=job_directory_path,
        blank=True,
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=["zip"])],
    )

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

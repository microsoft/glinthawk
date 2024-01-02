import uuid
import celery

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


def job_directory_path(instance, filename):
    return "raw/user_{0}/{1}".format(
        instance.created_by.id, str(instance.uuid) + ".zip"
    )


class Job(models.Model):
    class Status(models.IntegerChoices):
        SUBMITTED = 1, "Submitted"
        PROCESSING = 2, "Processing"
        PROCESSED = 3, "Processed"
        COMPLETED = 4, "Completed"
        FAILED = 5, "Failed"

    class LanguageModel(models.TextChoices):
        STORIES_110M = "stories-110m", "Stories 110M"
        LLAMA2_7B_CHAT = "llama2-7b-chat", "LLaMa2 7B Chat"
        LLAMA2_13B_CHAT = "llama2-13b-chat", "LLaMa2 13B Chat"
        LLAMA2_70B_CHAT = "llama2-70b-chat", "LLaMa2 70B Chat"

    uuid = models.UUIDField(
        unique=True, editable=False, primary_key=True, default=uuid.uuid4
    )

    language_model = models.CharField(
        max_length=32, choices=LanguageModel.choices, default=LanguageModel.LLAMA2_70B_CHAT
    )

    file = models.FileField(
        upload_to=job_directory_path,
        blank=True,
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=["zip"])],
    )

    status = models.IntegerField(choices=Status.choices, default=Status.SUBMITTED)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        celery.current_app.send_task(
            "prompts.tasks.prepare_prompts_for_job", args=[self.uuid], countdown=5
        )


class Prompt(models.Model):
    class Status(models.IntegerChoices):
        SUBMITTED = 1, "Submitted"
        PREPROCESSED = 2, "Preprocessed"
        QUEUED = 3, "Queued"
        STARTED = 4, "Started"
        COMPLETED = 5, "Completed"
        FAILED = 6, "Failed"

    uuid = models.UUIDField(
        unique=True, editable=False, primary_key=True, default=uuid.uuid4
    )

    job = models.ForeignKey(Job, on_delete=models.CASCADE, null=True)
    hash = models.CharField(max_length=64)
    status = models.IntegerField(choices=Status.choices, default=Status.SUBMITTED)
    preprocessed_at = models.DateTimeField(null=True, blank=True)
    queued_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

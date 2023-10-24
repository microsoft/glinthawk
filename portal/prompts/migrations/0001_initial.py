# Generated by Django 4.2.6 on 2023-10-24 21:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Job",
            fields=[
                (
                    "uuid",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                        unique=True,
                    ),
                ),
                (
                    "language_model",
                    models.CharField(
                        choices=[
                            ("llama2_7b", "LLaMa2 7B"),
                            ("llama2_7b_chat", "LLaMa2 7B Chat"),
                            ("llama2_70b", "LLaMa2 70B"),
                            ("llama2_70b_chat", "LLaMa2 70B Chat"),
                        ],
                        default="llama2_7b",
                        max_length=32,
                    ),
                ),
                ("file", models.FileField(blank=True, null=True, upload_to="raw/")),
                ("url", models.URLField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "created_by",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Prompt",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("hash", models.CharField(max_length=64)),
                (
                    "status",
                    models.IntegerField(
                        choices=[
                            (1, "Submitted"),
                            (2, "Preprocessing"),
                            (3, "Preprocessed"),
                            (4, "Completed"),
                        ],
                        default=1,
                    ),
                ),
                ("preprocessed_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "job",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="prompts.job",
                    ),
                ),
            ],
        ),
    ]

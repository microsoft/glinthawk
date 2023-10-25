# Generated by Django 4.2.6 on 2023-10-25 22:00

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):
    dependencies = [
        ("prompts", "0003_job_status_alter_prompt_status"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="prompt",
            name="id",
        ),
        migrations.AddField(
            model_name="prompt",
            name="uuid",
            field=models.UUIDField(
                default=uuid.uuid4,
                editable=False,
                primary_key=True,
                serialize=False,
                unique=True,
            ),
        ),
    ]

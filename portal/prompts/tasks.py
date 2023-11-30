from .models import Job, Prompt
from celery import shared_task

import os
import sys
import glob
import zipfile
import logging
import tempfile
import datetime
import traceback
import urllib.request
import multiprocessing as mp

from django.conf import settings as django_settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "py"))
from common.tokenizer import Tokenizer
from common import serdes

logging.basicConfig(level=logging.INFO)
tokenizer = Tokenizer(django_settings.GLINTHAWK_TOKENIZER_PATH)


def process_file(job, file_path, output_dir):
    # TODO(sadjad): add safeguards to prevent processing XL, invalid, etc. files
    global tokenizer

    with open(file_path, "r") as f:
        data = f.read()

    tokens = tokenizer.encode(data, prepend_bos=True)
    output_hash, serialized_data = serdes.serialize(tokens, output_dir)

    default_storage.save(f"processed/{output_hash}.ghp", ContentFile(serialized_data))

    prompt = Prompt.objects.create(job=job)
    prompt.hash = output_hash
    prompt.save()

    return prompt.uuid


@shared_task
def prepare_prompts_for_job(job_uuid):
    try:
        job = Job.objects.get(uuid=job_uuid)

        # create a temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(default_storage.open(job.file.name)) as zip_file:
                zip_file.extractall(temp_dir)

            files_to_proccess = []

            for f in os.listdir(temp_dir):
                if not f.endswith(".txt"):
                    continue

                files_to_proccess.append(os.path.join(temp_dir, f))

            # process files
            for f in files_to_proccess:
                process_file(job, f, temp_dir)

            job.status = Job.Status.PROCESSED
            job.preprocessed_at = datetime.datetime.now()
            job.save()

    except Exception as e:
        print(e)
        traceback.print_exc()
        if job:
            job.status = Job.Status.FAILED
            job.save()
        return

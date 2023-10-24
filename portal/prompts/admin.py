from django.contrib import admin

from .models import Prompt

class PromptAdmin(admin.ModelAdmin):
    pass

# Register your models here.
admin.site.register(Prompt, PromptAdmin)

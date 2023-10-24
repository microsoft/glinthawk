from django.contrib import admin

from .models import Job, Prompt

class PromptAdmin(admin.ModelAdmin):
    pass

class JobAdmin(admin.ModelAdmin):
    pass

# Register your models here.
admin.site.register(Job, JobAdmin)
admin.site.register(Prompt, PromptAdmin)

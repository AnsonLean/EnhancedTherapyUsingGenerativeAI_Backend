from django.db import models

# Create your models here.

class QueryExplainer(models.Model):
    _input = models.TextField()
    _output = models.TextField()

    class Meta:
        db_table = "t_query"

class AddDocumentModel(models.Model):
    _input = models.TextField()
    _output = models.TextField()

    class Meta:
        db_table = "t_add_document"

class TranscribeModel(models.Model):
    video_name = models.TextField()
    session_id = models.TextField()
    description = models.TextField()
    client = models.TextField()
    date = models.TextField()
    _output = models.TextField()

    class Meta:
        db_table = "t_transcribe"


class CodeExplainer(models.Model):
    _input = models.TextField()
    _output = models.TextField()

    class Meta:
        db_table = "t_code_explainer"
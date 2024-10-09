from django.urls import path
#internals
from app.views import QueryView, AddDocumentView, TranscribeView, CodeExplainView

urlpatterns = [
    path('topic-query/',QueryView.as_view(),name='topic-query'),
    path('add-documents/',AddDocumentView.as_view(),name='add-document'),
    path('transcribe/',TranscribeView.as_view(),name='transcribe-document'),
    path ('activity-query/', CodeExplainView.as_view(), name = 'activity-query'),
]

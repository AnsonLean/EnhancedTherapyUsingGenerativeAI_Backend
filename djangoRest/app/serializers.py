from rest_framework import serializers
#internal
from app.models import QueryExplainer, AddDocumentModel, TranscribeModel, CodeExplainer
from app.utils import send_query_to_api, add_document_to_db, transcribe_video, send_code_to_api

class QuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryExplainer
        fields = ['id','_input', '_output']
        extra_kwargs = {
            '_output': {"read_only": True},
        }

    def create(self, validated_data):
        qe = QueryExplainer(**validated_data)
        _output = send_query_to_api(validated_data["_input"])
        qe._output = _output
        qe.save()
        return qe
    
class AddDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = AddDocumentModel
        fields = ['id','_input', '_output']
        extra_kwargs = {
            '_output': {"read_only": True},
        }
        
    def create(self, validated_data):
        qe = AddDocumentModel(**validated_data)
        _output = add_document_to_db(validated_data["_input"])
        qe._output = _output
        qe.save()
        return qe
    
class TranscribeSerializer(serializers.ModelSerializer):
    class Meta:
        model = TranscribeModel
        fields = ['id','video_name', 'session_id', 'description', 'client', 'date', '_output']
        extra_kwargs = {
            '_output': {"read_only": True},
        }
        
    def create(self, validated_data):
        qe = TranscribeModel(**validated_data)
        _output = transcribe_video(validated_data["video_name"], validated_data["session_id"], validated_data["description"], validated_data["client"], validated_data["date"])
        qe._output = _output
        qe.save()
        return qe


class CodeExplainSerializer(serializers.ModelSerializer):
    class Meta:
        model = CodeExplainer
        fields = ("id", "_input", "_output")
        extra_kwargs = {
            "_output":{"read_only": True}
        }

    def create(self, validated_data):
        ce = CodeExplainer(**validated_data)
        _output = send_code_to_api(validated_data["_input"])
        ce._output = _output 
        ce.save()
        return ce

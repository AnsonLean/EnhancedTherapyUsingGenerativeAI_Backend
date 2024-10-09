from rest_framework import views, status
from rest_framework.response import Response

# internals
from app.serializers import QuerySerializer, AddDocumentSerializer, TranscribeSerializer, CodeExplainSerializer
from app.models import QueryExplainer, AddDocumentModel, TranscribeModel, CodeExplainer
from app.utils import delete_vector_db

class QueryView(views.APIView):
    serializer_class = QuerySerializer
    # authentication_classes = [TokenAuthentication]

    def get(self, request, format=None):
        qe = QueryExplainer.objects.all()
        serializer = self.serializer_class(qe, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(): 
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    #To reset db
    def delete(self, request, format=None):
        qe = QueryExplainer.objects.all()
        qe.delete()
        delete_vector_db()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
class AddDocumentView(views.APIView):
    serializer_class = AddDocumentSerializer

    def get(self, request, format=None):
        qe = AddDocumentModel.objects.all()
        serializer = self.serializer_class(qe, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(): 
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class TranscribeView(views.APIView):
    serializer_class = TranscribeSerializer

    def get(self, request, format=None):
        qe = TranscribeModel.objects.all()
        serializer = self.serializer_class(qe, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(): 
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CodeExplainView(views.APIView):
    serializer_class = CodeExplainSerializer

    def get(self, request, format = None):
        qs = CodeExplainer.objects.all()
        serializer = self.serializer_class(qs, many = True)
        return Response(serializer.data)

    def post(self, request, format = None):
        serializer = self.serializer_class(data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
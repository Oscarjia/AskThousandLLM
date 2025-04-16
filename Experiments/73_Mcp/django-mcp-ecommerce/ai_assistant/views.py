# In your terminal:
# python manage.py startapp ai_assistant

# ai_assistant/views.py
from django.http import JsonResponse
from django.shortcuts import render
import json
import uuid

def ai_assistant_view(request):
    """View for rendering the AI assistant chat interface."""
    return render(request, 'ai_assistant/chat_window.html')

def generate_response(request):
    """API endpoint to generate AI responses."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            messages = data.get('messages', [])
            
            # For demo purposes, just echo back the last message
            # In production, you would connect to an actual LLM API
            last_message = messages[-1]['content'] if messages else ""
            
            response = {
                'id': str(uuid.uuid4()),
                'role': 'assistant',
                'content': f"This is a demo response to: {last_message}",
                'parts': []
            }
            
            return JsonResponse(response)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

# ai_assistant/urls.py
from django.urls import path
from . import views

app_name = 'ai_assistant'

urlpatterns = [
    path('chat/', views.ai_assistant_view, name='chat'),
    path('generate/', views.generate_response, name='generate'),
]
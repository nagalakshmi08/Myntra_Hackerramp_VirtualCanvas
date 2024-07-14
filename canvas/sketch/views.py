from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from django.http import JsonResponse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50 
from sklearn.neighbors import NearestNeighbors
import pickle


@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image_data')
        if image_data:
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)
            print(image_array)
            # Process the image with your ML algorithm here

            # Return the result (for demonstration, we just return a success message)
            return JsonResponse({'message': 'Image processed successfully'})
    return JsonResponse({'error': 'Invalid request'}, status=400)

# Create your views here.
def home(request):
    return render(request,'nav.html')

def sketchpage(request):
    if request.method == 'POST':
        return render(request,'canva.html')
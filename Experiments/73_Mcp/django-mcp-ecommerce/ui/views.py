from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Product

# Create your views here.

def product_list(request):
    products = Product.objects.all()
    return render(request, 'products/product_list.html', {'products': products})

def product_detail(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    return render(request, 'products/product_detail.html', {'product': product})

# API Views to match frontend
def api_products(request):
    products = Product.objects.all()
    data = [{
        'id': product.id,
        'name': product.name,
        'image': product.image,
        'description': product.description,
        'shortDescription': product.short_description,
        'price': float(product.price)
    } for product in products]
    return JsonResponse(data, safe=False)

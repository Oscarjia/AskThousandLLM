from django.urls import path
from . import views

app_name = 'ui'

urlpatterns = [
    path('products/', views.product_list, name='product_list'),
    path('products/<int:product_id>/', views.product_detail, name='product_detail'),
    # API endpoint to match frontend
    path('api/products/', views.api_products, name='api_products'),
]   
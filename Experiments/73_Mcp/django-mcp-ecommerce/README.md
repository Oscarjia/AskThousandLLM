# Django MCP E-commerce

This is a Django implementation of a Model Context Protocol (MCP) for an e-commerce application.

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install django djangorestframework django-cors-headers
```

3. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

4. Create a superuser:
```bash
python manage.py createsuperuser
```

5. Run the development server:
```bash
python manage.py runserver
```

## API Endpoints

- `GET /api/products/` - Get all products
- `GET /api/products/{id}/` - Get a specific product
- `GET /api/products/get_products/` - Get all products (alternative endpoint)
- `GET /api/products/{id}/recommend_product/` - Get product recommendations

## Features

- RESTful API for product management
- Model Context Protocol implementation
- Product recommendation system (basic implementation)
- CORS enabled for frontend integration

## Frontend Integration

The API can be integrated with any frontend framework. The endpoints are CORS-enabled for easy integration. 
from django.db import models
from django.conf import settings
import os

# Create your models here.

class Product(models.Model):
    name = models.CharField(max_length=200)
    image = models.CharField(max_length=200)  # Changed from URLField to CharField
    description = models.TextField()
    short_description = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    @property
    def image_url(self):
        if self.image.startswith('http'):
            return self.image
        return f'/static/images/{self.image}'

    class Meta:
        ordering = ['-created_at']

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('objectives/', views.objectives, name='objectives'),
    path('predict/<str:model_name>/', views.predict, name='predict_model'),
    path('predict/', views.predict, name='predict_default'), # Default to model1
]

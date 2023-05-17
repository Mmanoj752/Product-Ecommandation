from django.urls import path
from django.contrib.auth.views import LogoutView
from .views import home, user_login, product_list,recommendation

app_name = 'eshope'

urlpatterns = [
    # path('', home, name='home'),
    path('login/', user_login, name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('', product_list, name='products'),
    path('recommendation/', recommendation, name='recommendation'),
]

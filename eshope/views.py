from django.shortcuts import render
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.contrib.auth.hashers import check_password
from django.contrib.auth import authenticate, login
from django.http import HttpResponseRedirect, JsonResponse
from surprise import *
from .models import Product, user_table
from collections import defaultdict


def product_list(request):
    products = Product.objects.all()
    paginator = Paginator(products, 30)  # Number of products per page

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'eshope/product_list.html', {'page_obj': page_obj})


def home(request):
    if request.method == 'POST':
        product_id = request.POST.get('product_id')

        # Get the recommended products based on the clicked product
        recommended_products = get_recommended_products(product_id)

        return render(request, 'eshope/product_list.html', {'recommended_products': recommended_products})

    # Handle other parts of the home page view
    # ...

    return render(request, 'eshope/product_list.html')


def user_login(request):
    if request.method == 'POST':
        user_id = request.POST.get('userid')
        password = request.POST.get('password')

        try:
            user = user_table.objects.get(userid=user_id)
        except user_table.DoesNotExist:
            # Handle the case when the user does not exist
            return render(request, 'login.html', {'error_message': 'Invalid credentials'})

        if check_password(password, user.password):
            # Authentication successful
            login(request, user)
            return HttpResponseRedirect('/products/')
        else:
            # Handle the case when the password is incorrect
            return render(request, 'login.html', {'error_message': 'Invalid credentials'})

    return render(request, 'login.html')


import joblib

svd_model = joblib.load('svd_model.pkl')
# or
svd_model = joblib.load('svdpp_model.pkl')


def get_recommended_products(product_id):
    # Use the SVD model to generate recommendations for the given product_id
    recommendations = svd_model.get_recommendations(product_id)

    # Convert the recommendations to the required format
    recommendation_list = [(product.uid, product.iid, product.r_ui, product.est, product.details) for product in recommendations]

    # Get the top n recommendations
    top_n_recommendations = get_top_n_recommendations(recommendation_list, n=5)

    # Retrieve the recommended products from the database or any other data source
    recommended_product_ids = [item[0] for item in top_n_recommendations]
    recommended_products = Product.objects.filter(id__in=recommended_product_ids)

    return recommended_products


def get_top_n_recommendations(recommendations, n=5):
    # First map the recommendations to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in recommendations:
        top_n[uid].append((iid, est))

    # Sort predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def product_detail(request, product_id):
    # Retrieve the recommended products for the given product_id
    recommended_products = get_recommended_products(product_id)

    context = {
        'recommended_products': recommended_products,
    }
    return render(request, 'product_detail.html', context)

def recommendation(request):
    svd_model = joblib.load('svd_model.pkl')
    product_id = request.GET.get('productID')
    user_id = request.GET.get('userID')
    pred = algo_svd.test([(user_id, product_id)])
    recommended_products = get_top_n_recommendations(pred,5)
    for uid, user_ratings in recommended_products.items():
        print(uid, [iid for (iid, _) in user_ratings])
    return render(request, 'recommendation.html', {'recommended_products': recommended_products})


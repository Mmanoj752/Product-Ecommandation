from django.db import models

class Product(models.Model):
    id = models.AutoField(primary_key=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    asin = models.CharField(max_length=50)
    imageurlhighres = models.URLField(blank=True)

    def __str__(self):
        return str(self.id)  # Use another field, such as 'id' or 'price'

    class Meta:
        db_table = 'products'


from django.contrib.auth.models import AbstractBaseUser

class user_table(AbstractBaseUser):
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=20, unique=True)
    password = models.CharField(max_length=128)  # Add the password field
    last_login = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'userid'  # Specify the field to be used as the username
    
    def __str__(self):
        return self.userid

    class Meta:
        db_table = 'user_table'

from django.core.files.storage import FileSystemStorage
from django.db import models
from django.contrib.auth.models import User
from django.utils.html import escape

# Create your models here.

fs = FileSystemStorage()

CLOTH_TYPE = (
    ('dress', 'Dress'),
    ('shirt', 'Shirt'),
    ('pants', 'Pants'),
    ('shorts', 'Shorts'),
    ('shoes', 'Shoes'),
)

GENDER = (
    ('Boys', 'Boys'),
    ('Men', 'Men'),
    ('Unisex', 'Unisex'),
    ('Women', 'Women'),
    ('Girls', 'Girls'),
)

USAGE = (
    ('Formal', 'Formal'),
    ('Smart Casual', 'Smart Casual'),
    ('Sports', 'Sports'),
    ('Party', 'Party'),
    ('Home', 'Home'),
    ('NA', 'NA'),
    ('Ethnic', 'Ethnic'),
    ('Casual', 'Casual'),
    ('Travel', 'Travel'),
    # ('', ''),
)

AGE_GROUP = (
    # ('', ''),
    ('Kids-Girls', 'Kids-Girls'),
    ('Kids-Unisex', 'Kids-Unisex'),
    ('Kids-Boys', 'Kids-Boys'),
    ('Adults-Women', 'Adults-Women'),
    ('Adults-Unisex', 'Adults-Unisex'),
    ('Adults-Men', 'Adults-Men'),
)

SEASONS = (
    # ('', ''),
    ('Fall', 'Fall'),
    ('Summer', 'Summer'),
    ('Winter', 'Winter'),
    ('Spring', 'Spring'),
)


class ImageObject(models.Model):
    name = models.TextField(null=True, blank=True)  # productDisplayName
    colour = models.CharField(max_length=50)
    # type = models.CharField(max_length=50, choices=CLOTH_TYPE)
    image = models.ImageField(blank=True, null=True)
    image_link_front = models.TextField(null=True, blank=True)
    image_link_back = models.TextField(null=True, blank=True)
    image_link_right = models.TextField(null=True, blank=True)
    image_link_left = models.TextField(null=True, blank=True)
    image_buy_link = models.TextField(null=True, blank=True)  # landingPageUrl

    is_custom = models.BooleanField(verbose_name='Object uploaded by user')
    # True: no link, limited options, will be saved as image

    gender = models.CharField(max_length=10, choices=GENDER, null=True, blank=True)
    age_group = models.CharField(max_length=25, choices=AGE_GROUP, null=True, blank=True)

    brand = models.CharField(max_length=100)
    season = models.CharField(max_length=20, choices=SEASONS, null=True, blank=True)
    usage = models.CharField(max_length=20, choices=USAGE, null=True, blank=True)

    main_category = models.CharField(max_length=60, null=True, blank=True)
    sub_category = models.CharField(max_length=60, null=True, blank=True)
    article_category = models.CharField(max_length=60, null=True, blank=True)

    # todo think about use uploading image, for colour, type and gender
    # 1. accept from user
    # 2. profit

    def save(self, *args, **kwargs):
        # comment this out if need name to be something diff, idk why tho (idk why name field there)
        # self.name = self.image.name
        super(ImageObject, self).save(*args, **kwargs)

    def delete(self, using=None, keep_parents=False):
        fs.delete(self.image.name)
        super().delete()

    def __str__(self):
        return f'{self.article_category} | {self.colour} | {self.name}'


class FeatureVector(models.Model):
    # vector is stored in bytes object via pickle. use loads to load it as array
    vector = models.BinaryField()
    image_link = models.ForeignKey(ImageObject, on_delete=models.CASCADE)
    array_datatype = models.CharField(max_length=10, default='float32')

    def __str__(self):
        return f'{self.image_link}'


class SimilarityMatrix(models.Model):
    column_item = models.ForeignKey(ImageObject, on_delete=models.CASCADE, related_name='column_item')
    row_item = models.ForeignKey(ImageObject, on_delete=models.CASCADE, related_name='row_item')
    value = models.FloatField()

    def __str__(self):
        col_name = self.column_item
        row_name = self.row_item
        return f'{self.column_item} X {self.row_item}'


class Wishlist(models.Model):
    item = models.ForeignKey(ImageObject, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.user.get_full_name()


class Review(models.Model):
    review = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    review_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if len(self.review) > 20:
            return self.review[:20] + '...'
        else:
            return self.review


class UserAvatarSocial(models.Model):
    social_pfp = models.TextField()
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.user.get_full_name()


class Metric(models.Model):
    article = models.CharField(max_length=100)
    recall = models.FloatField()
    precision = models.FloatField()
    f1 = models.FloatField()

    def __str__(self):
        return f'R:{self.recall} | P:{self.precision} | F:{self.f1}'


class RecHistory(models.Model):
    image = models.ForeignKey(ImageObject, on_delete=models.CASCADE, null=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.user}'

from django.contrib import admin

# Register your models here.
from django.utils.html import format_html

from home.models import *


@admin.register(ImageObject)
class ImageDBAdmin(admin.ModelAdmin):
    list_filter = ['colour', 'main_category', 'sub_category']
    list_display = ['name', 'gender', 'is_custom', 'colour', 'season', 'usage']

    temp_fields = [field.name for field in ImageObject._meta.get_fields(False, False)]
    temp_fields.append('image_tag')
    fields = tuple(temp_fields[5:])

    readonly_fields = ('image_tag',)

    def image_tag(self, obj):
        if obj.is_custom:
            return format_html('<img src="{}" />'.format(obj.image.url))
        else:
            return format_html('<img width="250px" src="{}" />'.format(obj.image_link_front))

    image_tag.short_description = 'Image'

    class Meta:
        model = ImageObject


admin.site.register(FeatureVector)
admin.site.register(SimilarityMatrix)
admin.site.register(Wishlist)
admin.site.register(Review)
admin.site.register(UserAvatarSocial)
admin.site.register(Metric)
admin.site.register(RecHistory)
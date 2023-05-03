from ..models import *

def update_user_pfp(strategy, *args, **kwargs):
    response = kwargs['response']
    backend = kwargs['backend']
    user = kwargs['user']

    if response['picture']:
        url = response['picture']
        if url:
            if not UserAvatarSocial.objects.filter(user=user):
                u = UserAvatarSocial(user=user, social_pfp=url)
                u.save()
            else:
                u = UserAvatarSocial.objects.filter(user=user)[0]
                u.social_pfp = url
                u.save()
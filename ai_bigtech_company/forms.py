from django import forms
from django.core.exceptions import ValidationError


def validate_keyword(value):
    if not value.isalpha():
        raise ValidationError('키워드는 문자만 입력해야 합니다.')


def validate_cluster_num(value):
    if not str(value).isdigit():
        raise ValidationError("ClusterNum에는 숫자만 입력할 수 있습니다.")

class MyForm(forms.Form):
    keyword = forms.CharField(max_length=20, validators=[validate_keyword])
    clusterNum = forms.IntegerField(validators=[validate_cluster_num])
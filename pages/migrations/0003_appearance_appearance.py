# Generated by Django 2.2 on 2022-04-07 20:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0002_auto_20220407_2055'),
    ]

    operations = [
        migrations.AddField(
            model_name='appearance',
            name='appearance',
            field=models.CharField(default='theme', max_length=10, unique=True),
        ),
    ]
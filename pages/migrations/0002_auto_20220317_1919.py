# Generated by Django 2.2 on 2022-03-17 19:19

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Action',
        ),
        migrations.DeleteModel(
            name='ModelSettings',
        ),
        migrations.DeleteModel(
            name='SecurityThreat',
        ),
    ]
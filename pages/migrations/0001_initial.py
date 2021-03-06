# Generated by Django 2.2 on 2022-04-07 20:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Appearance',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('theme', models.CharField(default='', max_length=10, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='DashboardModelSettings',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name_id', models.CharField(default='', max_length=100, unique=True)),
                ('setting', models.CharField(default='', max_length=100, unique=True)),
                ('switch', models.BooleanField(default=True, verbose_name='switch')),
            ],
        ),
        migrations.CreateModel(
            name='DashboardVideoSettings',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name_id', models.CharField(default='', max_length=100, unique=True)),
                ('setting', models.CharField(default='', max_length=100, unique=True)),
                ('switch', models.BooleanField(default=True, verbose_name='switch')),
            ],
        ),
    ]

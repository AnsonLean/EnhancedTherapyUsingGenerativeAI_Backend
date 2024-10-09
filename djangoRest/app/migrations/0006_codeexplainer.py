# Generated by Django 5.0.8 on 2024-09-27 13:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_transcribemodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='CodeExplainer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('_input', models.TextField()),
                ('_output', models.TextField()),
            ],
            options={
                'db_table': 't_code_explainer',
            },
        ),
    ]

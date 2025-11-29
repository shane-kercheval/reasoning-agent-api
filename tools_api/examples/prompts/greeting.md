---
name: greeting
description: Generate a personalized greeting message
category: communication
arguments:
  - name: name
    required: true
    description: Name of the person to greet
  - name: time_of_day
    required: false
    description: Time of day (morning, afternoon, evening) for contextual greeting
tags:
  - greeting
  - communication
---
{% if time_of_day == "morning" %}
Good morning, {{ name }}! I hope you have a wonderful day ahead.
{% elif time_of_day == "afternoon" %}
Good afternoon, {{ name }}! How is your day going?
{% elif time_of_day == "evening" %}
Good evening, {{ name }}! I hope you had a great day.
{% else %}
Hello, {{ name }}! It's nice to meet you.
{% endif %}

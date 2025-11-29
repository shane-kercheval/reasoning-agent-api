---
name: greeting
description: Generate a greeting message
category: example
arguments:
  - name: name
    required: true
    description: Person's name to greet
  - name: formal
    required: false
    description: Whether to use formal greeting
tags:
  - example
  - test
---
{% if formal %}Good day, {{ name }}. How may I assist you today?{% else %}Hey {{ name }}! What can I help you with?{% endif %}

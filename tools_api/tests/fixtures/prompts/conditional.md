---
name: conditional_test
description: A test prompt with conditional logic
category: test
arguments:
  - name: language
    required: true
    description: Programming language
  - name: focus
    required: false
    description: Specific focus areas
tags:
  - test
  - conditional
---
You are reviewing {{ language }} code.
{% if focus %}
Focus on: {{ focus }}
{% endif %}
Please provide detailed feedback.

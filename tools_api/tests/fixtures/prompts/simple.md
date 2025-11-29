---
name: simple_test
description: A simple test prompt with variable substitution
category: test
arguments:
  - name: name
    required: true
    description: Name to greet
tags:
  - test
  - simple
---
Hello, {{ name }}! This is a simple test prompt.

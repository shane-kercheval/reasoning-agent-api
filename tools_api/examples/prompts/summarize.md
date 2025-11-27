---
name: summarize
description: Summarize text content concisely
arguments:
  - name: max_sentences
    required: false
    description: Maximum number of sentences in the summary (default is 3)
---
Please provide a concise summary of the following content.
{% if max_sentences %}
Limit your summary to approximately {{ max_sentences }} sentences.
{% else %}
Limit your summary to approximately 3 sentences.
{% endif %}

Focus on the key points and main ideas. Be clear and direct.

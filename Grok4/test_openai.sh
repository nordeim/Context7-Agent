curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "developer",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'

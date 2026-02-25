import ollama

response = ollama.chat(
    model='phi3',
    messages=[
        {'role': 'user', 'content': 'Hello'}
    ]
)

print(response['message']['content'])

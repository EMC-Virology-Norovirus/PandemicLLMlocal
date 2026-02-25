import ollama, json, traceback
try:
    resp = ollama.chat(model='mistral', messages=[{'role':'user','content':'Hello from test script'}])
    print(json.dumps(resp, indent=2))
except Exception as e:
    traceback.print_exc()
    print('EXCEPTION_STR:', repr(e))

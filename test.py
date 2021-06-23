import requests

# resp = requests.post("http://127.0.0.1:5000/caption")
# print(resp.text)

files = {'file': open('test.mp4', 'rb')}
resp = requests.post("http://127.0.0.1:5000/upload", files=files)
print(resp.text)
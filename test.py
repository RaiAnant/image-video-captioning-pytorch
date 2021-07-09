import requests

# resp = requests.post("http://127.0.0.1:5000/caption")
# # print(resp.text)
#
# files = {'file': open('test.mp4', 'rb')}
# resp = requests.post("http://127.0.0.1:5000/upload", files=files)
# print(resp.text)

resp = requests.post("http://127.0.0.1:5000/caption", params = {"id":"1625887453.941216"})
print(resp.text)
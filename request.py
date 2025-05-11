import base64, requests

with open("download.jpeg", "rb") as f:
    img_str = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image": img_str,
    "prompt": "You are a social media influencer. Write a captivating Instagram caption for this image."
}

response = requests.post("http://129.114.108.40:8000/generate_caption", json=payload)
#response = requests.post("http://localhost:8000/generate_caption", json=payload)

print(response.json())
~                               
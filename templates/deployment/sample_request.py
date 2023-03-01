import requests

# Mock data
body = {
    "data": [
        {"x1": 1.0, "x2": -1.0},
        {"x1": -1.0, "x2": 1.0},
    ]
}

# Create a post request to check the web server.
local_server_address = "http://0.0.0.0:8000/score"
response = requests.post(
    local_server_address,
    json=body,
    headers={"Content-Type": "application/json"},
)
assert response.status_code == 200, "Response failed!"
print(response.json())

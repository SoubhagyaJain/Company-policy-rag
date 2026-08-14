import requests
def test_chat():
    url = "http://localhost:8000/api/chat/stream"
    payload = {
        "message": "what is Ministerial Action/Purely Administrative action",
        "session_id": "test_session_12345",
        "model": "qwen2.5:7b"
    }
    print("Sending request...", flush=True)
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'), flush=True)
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    test_chat()

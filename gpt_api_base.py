import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt


openai.api_base = ""
openai.api_key = ""

api_base = ""
api_key = ""

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai(message,n=1,temperature=0.8,max_tokens=4096):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    )
    if n==1:
        return response["choices"][0]["message"]["content"]
    else:
        return [result["message"]["content"] for result in response["choices"]]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_api(messages, temperature=0.8,max_tokens=2048):
    url = api_base
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
    data = {
    "model": "gpt-3.5-turbo",
    "messages": messages,
    "max_tokens":max_tokens,
    "temperature":temperature
}
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    reply = response_data['choices'][0]['message']['content']
    total_tokens = response_data['usage']['total_tokens']
    return reply


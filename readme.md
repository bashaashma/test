import argparse
from dotenv import load_dotenv
from models import MODELS

load_dotenv()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="claude", choices=list(MODELS.keys()))
    args = p.parse_args()

    model = MODELS[args.model]()
    name = args.model
    print(f"\n[Model: {name}]  'exit' to quit, '/switch <name>', '/reset' to clear history")
    print(f"Available: {', '.join(MODELS.keys())}\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == "/reset":
            model.reset()
            print("  history cleared\n")
            continue
        if user_input.startswith("/switch "):
            new = user_input.split(" ", 1)[1].strip()
            if new not in MODELS:
                print(f"  unknown. choices: {list(MODELS)}\n")
                continue
            model = MODELS[new]()
            name = new
            print(f"  switched to {name} (fresh history)\n")
            continue
        try:
            print(f"bot ({name}) > {model.chat(user_input)}\n")
        except Exception as e:
            print(f"  error: {e}\n")

if __name__ == "__main__":
    main()




AWS_REGION=us-east-1

python chat.py --model claude


from bedrock_agentcore.runtime import BedrockAgentCoreApp
from models import MODELS

app = BedrockAgentCoreApp()
model = MODELS["claude"]()  # pick whichever for "production"

@app.entrypoint
def invoke(payload, context):
    prompt = payload.get("prompt", "")
    return {"response": model.chat(prompt)}

if __name__ == "__main__":
    app.run()



curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'



import os, boto3

class BedrockModel:
    """Talks to any Bedrock model via the Converse API. Maintains chat history."""
    def __init__(self, model_id: str, system_prompt: str = "You are a helpful assistant."):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        self.model_id = model_id
        self.system = [{"text": system_prompt}]
        self.history = []

    def chat(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": [{"text": prompt}]})
        resp = self.client.converse(
            modelId=self.model_id,
            messages=self.history,
            system=self.system,
            inferenceConfig={"maxTokens": 2048, "temperature": 0.7},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        self.history.append({"role": "assistant", "content": [{"text": text}]})
        return text

    def reset(self):
        self.history = []

CODER_PROMPT = (
    "You help the user with questions and code. "
    "When asked for code, give clean runnable examples with a brief explanation. "
    "Be concise."
)

MODELS = {
    "claude":   lambda: BedrockModel("us.anthropic.claude-sonnet-4-20250514", CODER_PROMPT),
    "haiku":    lambda: BedrockModel("us.anthropic.claude-haiku-4-5-20251001", CODER_PROMPT),
    "opus":     lambda: BedrockModel("us.anthropic.claude-opus-4-1-20250805", CODER_PROMPT),
    "llama":    lambda: BedrockModel("us.meta.llama3-3-70b-instruct-v1:0", CODER_PROMPT),
    "nova-pro": lambda: BedrockModel("us.amazon.nova-pro-v1:0", CODER_PROMPT),
}



import json, os, boto3

def get_secret(secret_name: str, key: str) -> str:
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_name)
    return json.loads(resp["SecretString"])[key]

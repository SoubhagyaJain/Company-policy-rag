from src.config import Settings
from llama_index.llms.ollama import Ollama

def test():
    settings = Settings()
    print("Model:", settings.llm_model)
    llm = Ollama(model=settings.llm_model, base_url=settings.ollama_base_url)
    resp = llm.complete("Hello, are you gemma?")
    print("Response:", resp)

if __name__ == "__main__":
    test()

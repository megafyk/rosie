import aisuite
from aisuite.provider import Provider
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

client = aisuite.Client()


class LocalProvider(Provider):
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.temperature = kwargs.get("temperature", 0.2)
        self.do_sample = kwargs.get("do_sample", True)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        self.return_full_text = kwargs.get("return_full_text", False)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)

    def chat_completions_create(self, model, messages):
        return self.generate(model, messages)

    def generate(self, model, message):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_id = AutoModelForCausalLM.from_pretrained(model)
        from transformers import pipeline
        pipeline = pipeline(
            model=model_id,
            tokenizer=tokenizer,
            device=self.device,
            task="text-generation",
            temperature=self.temperature,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            return_full_text=self.return_full_text,
            max_new_tokens=self.max_new_tokens,
        )

        llm = HuggingFacePipeline(pipeline=pipeline)
        response = llm.invoke(message)
        return response


def generate(model: str, messages):
    if model.startswith("local"):
        # local
        provider = LocalProvider()
        response = provider.chat_completions_create(model, messages)
    else:
        # aisuite
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    return response

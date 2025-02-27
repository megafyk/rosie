import aisuite
from aisuite.provider import Provider
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TEST_PART_CONTEXT, LLM_MODEL_TOKEN, TEST_PART_QUESTION, OPENROUTER_API_KEY

client_aisuite = aisuite.Client()
client_open_router = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def messages_to_prompt(messages):
    context = ""
    question = ""
    system = ""
    for message in messages:
        if message["role"] == "user":
            question = message["content"]
        elif message["role"] == "assistant":
            context = message["content"]
        elif message["role"] == "system":
            system = message["content"]

    template = """{test_part_system}

    {test_part_context} 
    {context}
    
    {test_part_question}
    {question}
    """

    prompt_template = PromptTemplate(
        input_variables=[
            "test_part_system",
            "test_part_context",
            "context",
            "test_part_question",
            "question"
        ],
        template=template
    )

    prompt = prompt_template.format(
        test_part_system=system,
        test_part_context=TEST_PART_CONTEXT,
        context=context,
        test_part_question=TEST_PART_QUESTION,
        question=question
    )

    return prompt


class LocalProvider(Provider):
    def __init__(self, **kwargs):
        login(token=LLM_MODEL_TOKEN)
        self.device = kwargs.get("device", "cpu")
        self.temperature = kwargs.get("temperature", 0.2)
        self.do_sample = kwargs.get("do_sample", True)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.1)
        self.return_full_text = kwargs.get("return_full_text", False)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)

    def chat_completions_create(self, model, messages):
        return self.generate(model, messages_to_prompt(messages))

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


def generate(model: str, messages: list, kwargs):
    if model.startswith("local"):
        model = model.split(":")[1]
        # local
        provider = LocalProvider(**kwargs)
        response = provider.chat_completions_create(model, messages)
    else:
        # aisuite
        response = client_aisuite.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    return response


def generate_openrouter(model, messages, kwargs):
    response = client_open_router.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response

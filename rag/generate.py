import aisuite
from config import (
    LLM_MODEL_TOKEN,
    OPENROUTER_API_KEY,
    TEST_PART_CONTEXT,
    TEST_PART_QUESTION,
)
from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as create_pipeline

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


class ChatCompletion:
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class Choice:
        def __init__(self, message, index=0, finish_reason="stop"):
            self.message = message
            self.index = index
            self.finish_reason = finish_reason

    def __init__(self, content):
        self.choices = [
            self.Choice(
                self.Message("assistant", content)
            )
        ]

class LocalProvider:
    _pipelines = {}  # Class variable to store pipelines for reuse

    def __init__(self):
        login(token=LLM_MODEL_TOKEN)

    def chat_completions_create(self, model, messages, kwargs):
        return self.generate(model, messages_to_prompt(messages), kwargs)

    def generate(self, model, message, kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_id = AutoModelForCausalLM.from_pretrained(model)

        # Extract generation kwargs from pipeline creation kwargs
        generation_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ['temperature', 'max_new_tokens', 'device']}

        pipeline_instance = create_pipeline(
            model=model_id,
            tokenizer=tokenizer,
            task="text-generation",
            **kwargs
        )
        llm = HuggingFacePipeline(pipeline=pipeline_instance)

        # Extract generation-specific kwargs
        generation_kwargs = {k: v for k, v in kwargs.items()
                            if k in ['temperature', 'max_new_tokens']}

        # Apply generation parameters dynamically
        if generation_kwargs:
            llm = llm.bind(**generation_kwargs)

        response = llm.invoke(message)

        return ChatCompletion(response)


def generate(model: str, messages: list, kwargs):
    if model.startswith("local"):
        model = model.split(":")[1]
        # local
        provider = LocalProvider()
        response = provider.chat_completions_create(model, messages, kwargs)
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

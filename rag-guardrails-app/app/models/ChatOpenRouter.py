# from langchain_openai import ChatOpenAI

# from typing import Optional
# from pydantic import Field, SecretStr

# import os

# class ChatOpenRouter(ChatOpenAI):
#     openai_api_key: Optional[SecretStr] = Field(
#         alias="api_key",
#         default_factory=os.environ.get("OPENROUTER_API_KEY", default=None),
#     )
#     @property
#     def lc_secrets(self) -> dict[str, str]:
#         return {"openai_api_key": "OPENROUTER_API_KEY"}

#     def __init__(self,
#                  openai_api_key: Optional[str] = None,
#                  **kwargs):
#         openai_api_key = (
#             openai_api_key or os.environ.get("OPENROUTER_API_KEY")
#         )
#         super().__init__(
#             base_url="https://openrouter.ai/api/v1",
#             openai_api_key=openai_api_key,
#             **kwargs
#         )

from typing import Any, Optional, List, Dict, Iterator, Union
import json
import os
from requests import request
from fastapi import HTTPException
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
import aiohttp
# from langchain_core.language_models.llms import BaseLLM

# Import Nemo Guardrails LLM Provider if needed
# from nemoguardrails.llm.providers.base import LLMProvider
from typing import Any, List, Optional, Dict, Iterator, Union
from pydantic import Field
import json
import os
from requests import request
from fastapi import HTTPException
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage
from langchain_core.callbacks.base import BaseCallbackManager



class ChatOpenRouter(BaseLanguageModel):
    """
    A custom LLM provider that integrates with OpenRouter.
    Full compatibility with LangChain's BaseLanguageModel.
    """

    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/"
    temperature: float = Field(default=0.7)  # Default value
    max_tokens: int = Field(default=256)     # Default value

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: Union[str, any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
                        ) -> str:
        """Synchronous call to the OpenRouter API."""
        # Unwrap StringPromptValue if present
        try:
            prompt = prompt.text  # Try .text (used in some LangChain versions)
        except AttributeError:
            try:
                prompt = prompt.content  # Try .content (used in others)
            except AttributeError:
                pass  # Leave as-is if not a prompt object

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
            "X-Title": "Your Application Name"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        response = request(
            method="POST",
            url=f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Asynchronous API call (not implemented for now)."""
        raise NotImplementedError("Streaming or async not implemented")
    
    async def agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> LLMResult:
            """Asynchronous generation from chat messages."""
            generations = []
            for msg in messages:
                prompt = msg.content

                # Unwrap PromptValue if present
                if isinstance(prompt, PromptValue):
                    prompt = prompt.text  # or prompt.content

                content = await self._async_call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                generations.append([GenerationChunk(text=content)])
            return LLMResult(generations=generations)


    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Generate results for multiple prompts."""
        generations = []
        for prompt in prompts:
            content = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([GenerationChunk(text=content)])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[GenerationChunk]:
        """Stream not implemented."""
        raise NotImplementedError("Streaming not supported")

    def get_num_tokens(self, text: str) -> int:
        """Estimate vailable token count (not implemented)."""
        return len(text.split())  

    def get_token_ids(self, text: str) -> List[int]:
        """Return token IDs (not implemented)."""
        return [hash(text)] 

    def get_all_token_ids(self, text: str) -> Dict[str, List[int]]:
        """List of token IDs per token (not implemented)."""
        raise NotImplementedError("Detailed token info not implemented")

    def get_callback_manager(self) -> Any:
        """Callback manager (not implemented)."""
        return None

    def run(self, *args, **kwargs):
        """Allow transformation of inputs."""
        return self._call(*args, **kwargs)

    def invoke(self, input: str, **kwargs) -> str:
        """Invoke the model with input string."""
        return self._call(input, **kwargs)

    def predict(self, input: str, **kwargs) -> str:
        """Predict with input string."""
        return self._call(input, **kwargs)

    def predict_messages(self, messages: List[Dict], **kwargs) -> str:
        """Predict message input."""
        return self._call(
            prompt="".join([msg["content"] for msg in messages]),
            **kwargs
        )

    def generate_prompt(self, prompts: List[str], **kwargs) -> LLMResult:
        """Generate from input prompts."""
        return self._generate(prompts, **kwargs)

    async def agenerate_prompt(
            self,
            prompts: List[Union[str, PromptValue]],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> LLMResult:
            """Asynchronous prompt generation."""
            generations = []
            for prompt in prompts:
                # Unwrap PromptValue if present
                if isinstance(prompt, PromptValue):
                    prompt = prompt.text  # or prompt.content
                content = await self._async_call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                generations.append([GenerationChunk(text=content)])
            return LLMResult(generations=generations)


    async def apredict(self, input: str, **kwargs) -> str:
        """Asynchronous predict."""
        result = await self.agenerate([input], **kwargs)
        return result.generations[0][0].text

    async def apredict_messages(
            self,
            messages: List[Dict],
            **kwargs: Any
        ) -> str:
            """Asynchronously predict messages."""
            prompt = " ".join([msg.get("content", "") for msg in messages])
            
            # Unwrap PromptValue if present
            if isinstance(prompt, PromptValue):
                prompt = prompt.text  # or prompt.content

            return await self._async_call(prompt, **kwargs)


    async def _async_call(
            self,
            prompt: Union[str, PromptValue],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Asynchronous call to the OpenRouter API."""
            # Unwrap PromptValue if present
            if isinstance(prompt, PromptValue):
                prompt = prompt.text  # or prompt.content

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
                "X-Title": "Your Application Name"
            }

            # Filter out non-serializable or unnecessary kwargs
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if not (isinstance(v, BaseCallbackManager) or "callback" in k.lower())
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **filtered_kwargs  # Only include safe, serializable kwargs
            }
            print(f"Payload: {payload}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        print(f"Error: {response.text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=await response.text()
                        )
                    print(f"Response: {response}")
                    result = await response.json()
                    print(f"Result: {result}")
                    return result["choices"][0]["message"]["content"]





    # def __init__(self, api_key: str, model: str, **kwargs):
    #     super().__init__(**kwargs)
    #     self.api_key = api_key
    #     self.model = model


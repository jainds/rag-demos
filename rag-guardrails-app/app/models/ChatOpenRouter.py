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
import aiohttp
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
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ChatOpenRouterConfig(BaseModel):
    """Configuration for ChatOpenRouter."""
    model: str
    api_key: str
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1000)
    top_p: float = Field(default=0.95)
    frequency_penalty: float = Field(default=0)
    presence_penalty: float = Field(default=0)

    class Config:
        arbitrary_types_allowed = True

class ChatOpenRouter(BaseLanguageModel):
    """
    A custom LLM provider that integrates with OpenRouter.
    Full compatibility with LangChain's BaseLanguageModel and Ragas metrics.
    """
    
    model: str = Field(default="opengvlab/internvl3-2b:free", description="The model identifier to use with OpenRouter")
    api_key: Optional[str] = Field(default=None, description="API key for OpenRouter")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate")
    top_p: float = Field(default=0.95, description="Top p sampling parameter")
    frequency_penalty: float = Field(default=0, description="Frequency penalty parameter")
    presence_penalty: float = Field(default=0, description="Presence penalty parameter")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="Base URL for OpenRouter API")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        """Initialize the ChatOpenRouter."""
        super().__init__(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            base_url=base_url,
            **kwargs
        )

    def _get_api_key(self) -> str:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set.")
            raise ValueError("OPENROUTER_API_KEY not set in environment or config.")
        return api_key

    def set_run_config(self, run_config=None, **kwargs):
        """Set run configuration for the model."""
        try:
            if run_config is not None:
                config_dict = getattr(run_config, "__dict__", {})
                config_dict.update(kwargs)
            else:
                config_dict = kwargs
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            logger.warning(f"Error in set_run_config: {str(e)}")
        return self

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _parse_response(self, response) -> str:
        try:
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError("No choices in OpenRouter response")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Malformed OpenRouter response: {getattr(response, 'text', str(response))}")
            raise

    def _call(
        self,
        prompt: Union[str, Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        logger.debug(f"ChatOpenRouter._call called with prompt: {prompt}")
        try:
            prompt = getattr(prompt, 'text', getattr(prompt, 'content', prompt))
            api_key = self._get_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
                "X-Title": "Your Application Name"
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, (BaseCallbackManager, CallbackManagerForLLMRun))}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": str(prompt)}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **filtered_kwargs
            }
            response = request(
                method="POST",
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                logger.error(f"OpenRouter API error ({response.status_code}): {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
            return self._parse_response(response)
        except Exception as e:
            logger.exception("Error in ChatOpenRouter._call")
            raise

    def _ensure_llmresult(self, result):
        if isinstance(result, LLMResult):
            return result
        if isinstance(result, str):
            return LLMResult(generations=[[GenerationChunk(text=result)]])
        raise ValueError(f"Unexpected LLM result type: {type(result)}")

    async def _async_call(
        self,
        prompt: Union[str, PromptValue],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        logger.debug(f"ChatOpenRouter._async_call called with prompt: {prompt}")
        try:
            prompt = getattr(prompt, 'text', getattr(prompt, 'content', prompt))
            api_key = self._get_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
                "X-Title": "Your Application Name"
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, (BaseCallbackManager, AsyncCallbackManagerForLLMRun))}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": str(prompt)}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **filtered_kwargs
            }
            import asyncio
            max_retries = 5
            for attempt in range(max_retries):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    ) as response:
                        text = await response.text()
                        if response.status == 429 or (text and 'rate limit' in text.lower()):
                            if attempt < max_retries - 1:
                                logger.warning(f"[OpenRouter] Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting 60 seconds before retry...")
                                await asyncio.sleep(60)
                                continue
                            else:
                                raise HTTPException(status_code=429, detail=f"OpenRouter API rate limit exceeded after {max_retries} retries: {text}")
                        if response.status != 200:
                            logger.error(f"OpenRouter API error ({response.status}): {text}")
                            raise HTTPException(
                                status_code=response.status,
                                detail=text
                            )
                        result = await response.json()
                        if not result or "choices" not in result or not result["choices"]:
                            if "error" in result and "rate limit" in str(result["error"]).lower():
                                if attempt < max_retries - 1:
                                    logger.warning(f"[OpenRouter] Rate limit error in JSON (attempt {attempt+1}/{max_retries}). Waiting 60 seconds before retry...")
                                    await asyncio.sleep(60)
                                    continue
                                else:
                                    raise HTTPException(status_code=429, detail=f"OpenRouter API rate limit exceeded after {max_retries} retries: {result}")
                            raise ValueError(f"Malformed response from OpenRouter: {result}")
                        logger.debug(f"ChatOpenRouter._async_call returning type: {type(result['choices'][0]['message']['content'])}")
                        return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception("Error in ChatOpenRouter._async_call")
            raise

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
        return self._ensure_llmresult(LLMResult(generations=generations))

    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResult:
        logger.debug(f"ChatOpenRouter.generate called with prompt: {prompt}")
        content = await self._async_call(prompt, **kwargs)
        logger.debug(f"ChatOpenRouter.generate _async_call returned type: {type(content)}")
        return self._ensure_llmresult(content)

    async def agenerate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResult:
        logger.debug(f"ChatOpenRouter.agenerate called with prompt: {prompt}")
        content = await self._async_call(prompt, **kwargs)
        logger.debug(f"ChatOpenRouter.agenerate _async_call returned type: {type(content)}")
        return self._ensure_llmresult(content)

    def get_num_tokens(self, text: str) -> int:
        """Estimate available token count."""
        return len(text.split())  

    def get_token_ids(self, text: str) -> List[int]:
        """Return token IDs."""
        return [hash(text)] 

    def predict(self, input: str, **kwargs) -> LLMResult:
        logger.debug(f"ChatOpenRouter.predict called with input: {input}")
        content = self._call(input, **kwargs)
        logger.debug(f"ChatOpenRouter.predict _call returned type: {type(content)}")
        return self._ensure_llmresult(content)

    def invoke(self, input: str, **kwargs) -> LLMResult:
        logger.debug(f"ChatOpenRouter.invoke called with input: {input}")
        result = self.predict(input, **kwargs)
        logger.debug(f"ChatOpenRouter.invoke returning type: {type(result)}")
        return result

    def predict_messages(self, messages: List[Dict], **kwargs) -> LLMResult:
        logger.debug(f"ChatOpenRouter.predict_messages called with messages: {messages}")
        content = self._call(
            prompt="".join([msg["content"] for msg in messages]),
            **kwargs
        )
        logger.debug(f"ChatOpenRouter.predict_messages _call returned type: {type(content)}")
        return self._ensure_llmresult(content)

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
            return self._ensure_llmresult(LLMResult(generations=generations))

    async def apredict(self, input: str, **kwargs) -> LLMResult:
        logger.debug(f"ChatOpenRouter.apredict called with input: {input}")
        result = await self.agenerate(input, **kwargs)
        logger.debug(f"ChatOpenRouter.apredict agenerate returned type: {type(result)}")
        return self._ensure_llmresult(result)

    async def apredict_messages(self, messages: List[Dict], **kwargs) -> LLMResult:
        logger.debug(f"ChatOpenRouter.apredict_messages called with messages: {messages}")
        prompt = " ".join([msg.get("content", "") for msg in messages])
        if isinstance(prompt, PromptValue):
            prompt = prompt.text
        content = await self._async_call(prompt, **kwargs)
        logger.debug(f"ChatOpenRouter.apredict_messages _async_call returned type: {type(content)}")
        return self._ensure_llmresult(content)

    async def agenerate_text(
        self,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """
        Asynchronously generate text from a prompt.
        This method is required by Ragas metrics.
        """
        try:
            prompt = str(prompt)  # Ensure prompt is a string
            headers = {
                "Authorization": f"Bearer {self._get_api_key()}",
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
                "X-Title": "Your Application Name"
            }

            # Filter out non-serializable kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                            if not isinstance(v, (BaseCallbackManager, AsyncCallbackManagerForLLMRun))}

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **filtered_kwargs
            }

            import asyncio
            max_retries = 5
            for attempt in range(max_retries):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    ) as response:
                        text = await response.text()
                        if response.status == 429 or (text and 'rate limit' in text.lower()):
                            if attempt < max_retries - 1:
                                print(f"[OpenRouter] Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting 60 seconds before retry...")
                                await asyncio.sleep(60)
                                continue
                            else:
                                raise HTTPException(status_code=429, detail=f"OpenRouter API rate limit exceeded after {max_retries} retries: {text}")
                        if response.status != 200:
                            raise HTTPException(
                                status_code=response.status,
                                detail=text
                            )
                        result = await response.json()
                        if not result or "choices" not in result or not result["choices"]:
                            # Check for rate limit in error JSON
                            if "error" in result and "rate limit" in str(result["error"]).lower():
                                if attempt < max_retries - 1:
                                    print(f"[OpenRouter] Rate limit error in JSON (attempt {attempt+1}/{max_retries}). Waiting 60 seconds before retry...")
                                    await asyncio.sleep(60)
                                    continue
                                else:
                                    raise HTTPException(status_code=429, detail=f"OpenRouter API rate limit exceeded after {max_retries} retries: {result}")
                            raise ValueError(f"Malformed response from OpenRouter: {result}")
                        return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error in agenerate_text: {str(e)}")
            return ""

    # def __init__(self, api_key: str, model: str, **kwargs):
    #     super().__init__(**kwargs)
    #     self.api_key = api_key
    #     self.model = model


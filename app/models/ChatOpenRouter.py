from typing import Any, Optional, List, Dict, Union
import os
import logging
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.prompt_values import PromptValue
from langchain_core.outputs import GenerationChunk, LLMResult
from fastapi import HTTPException
import aiohttp
from requests import request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

def _get_prompt_str(prompt):
    if hasattr(prompt, "text"):
        return prompt.text
    elif hasattr(prompt, "content"):
        return prompt.content
    elif hasattr(prompt, "to_string"):
        return prompt.to_string()
    else:
        return str(prompt)

def _call(
    self,
    prompt: Union[str, Any],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> str:
    logger.debug(f"ChatOpenRouter._call called with prompt: {prompt}")
    try:
        prompt_str = _get_prompt_str(prompt)
        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
            "X-Title": "Your Application Name"
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, (BaseCallbackManager, CallbackManagerForLLMRun))}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt_str)}],
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

async def _async_call(
    self,
    prompt: Union[str, PromptValue],
    stop: Optional[List[str]] = None,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> str:
    logger.debug(f"ChatOpenRouter._async_call called with prompt: {prompt}")
    try:
        prompt_str = _get_prompt_str(prompt)
        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8000"),
            "X-Title": "Your Application Name"
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, (BaseCallbackManager, AsyncCallbackManagerForLLMRun))}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt_str)}],
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
        prompt_str = _get_prompt_str(prompt)
        content = await self._async_call(prompt_str, stop=stop, run_manager=run_manager, **kwargs)
        generations.append([GenerationChunk(text=content)])
    return self._ensure_llmresult(LLMResult(generations=generations))

# Apply the same _get_prompt_str logic in any other method that handles prompts 
from typing import Any, List, Union, Optional, Dict
from langchain.schema import LLMResult
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.prompts import PromptValue

class ChatOpenRouter:
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResult:
        result = await self._async_call(prompt, **kwargs)
        return self._ensure_llmresult(result)

    async def agenerate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResult:
        result = await self._async_call(prompt, **kwargs)
        return self._ensure_llmresult(result)

    async def agenerate_prompt(
        self,
        prompts: List[Union[str, PromptValue]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> LLMResult:
        # For each prompt, call _async_call and wrap result
        generations = []
        for prompt in prompts:
            result = await self._async_call(prompt, **kwargs)
            llm_result = self._ensure_llmresult(result)
            generations.append(llm_result.generations[0])
        return LLMResult(generations=generations)

    async def apredict(self, input: str, **kwargs) -> LLMResult:
        result = await self._async_call(input, **kwargs)
        return self._ensure_llmresult(result)

    async def apredict_messages(self, messages: List[Dict], **kwargs) -> LLMResult:
        # Compose prompt from messages
        prompt = "\n".join([m.get("content", "") for m in messages])
        result = await self._async_call(prompt, **kwargs)
        return self._ensure_llmresult(result)

    async def agenerate_text(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMResult:
        result = await self._async_call(prompt, **kwargs)
        return self._ensure_llmresult(result) 
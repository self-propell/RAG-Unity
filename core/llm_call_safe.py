"""
Enhanced LLM call with timeout and retry

Wraps LLM provider calls with error handling.
"""
import time
from typing import Optional
from core.error_handling import FallbackStrategy, get_timeout, get_retry_count


def call_llm_with_retry(
    provider,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 4096,
    timeout: Optional[int] = None
):
    """
    Call LLM with timeout and retry.

    Args:
        provider: LLM provider instance
        system_prompt: System prompt
        messages: Message history
        tools: Available tools
        max_tokens: Max tokens to generate
        timeout: Timeout in seconds (default from config)

    Returns:
        ChatResult or raises exception
    """
    if timeout is None:
        timeout = get_timeout("llm_call")

    max_retries = get_retry_count("llm_call")

    last_error = None
    for attempt in range(max_retries):
        try:
            # TODO: Add actual timeout mechanism
            # For now, just call the provider
            result = provider.chat(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens
            )
            return result

        except TimeoutError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise

        except ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

        except Exception as e:
            # For other errors, don't retry
            raise

    # Should not reach here, but just in case
    raise last_error if last_error else Exception("LLM call failed")


def call_llm_safe(
    provider,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 4096,
    timeout: Optional[int] = None
):
    """
    Safe LLM call that never raises, always returns a result.

    Returns:
        (success: bool, result: ChatResult or error_message: str)
    """
    try:
        result = call_llm_with_retry(
            provider=provider,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            timeout=timeout
        )
        return True, result

    except TimeoutError as e:
        error_msg = FallbackStrategy.llm_timeout_fallback()
        return False, error_msg

    except Exception as e:
        error_msg = FallbackStrategy.llm_error_fallback(e)
        return False, error_msg

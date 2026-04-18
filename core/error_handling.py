"""
Enhanced error handling and timeout utilities for tool execution

Provides:
- Timeout decorators
- Retry mechanisms
- Error loop detection
- Fallback strategies
"""
import time
import signal
from functools import wraps
from typing import Callable, Any, Optional


class ToolTimeoutError(Exception):
    """Tool execution timeout"""
    pass


class ToolRetryError(Exception):
    """Tool retry exhausted"""
    pass


def with_timeout(timeout_seconds: int = 30):
    """
    Decorator to add timeout to tool functions.

    Args:
        timeout_seconds: Maximum execution time

    Returns:
        Decorated function that raises ToolTimeoutError on timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise ToolTimeoutError(f"Tool execution timeout ({timeout_seconds}s)")

            # Set timeout alarm (Unix only)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                return result

            except ToolTimeoutError as e:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return {"error": str(e), "timeout": True}

            except Exception as e:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return {"error": f"Execution failed: {e}"}

        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to add retry logic to tool functions.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # If result has timeout error and we have retries left, retry
                    if isinstance(result, dict) and result.get("timeout") and attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                        continue

                    return result

                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                        continue
                    else:
                        raise ToolRetryError(f"Failed after {max_attempts} attempts: {last_error}")

            return {"error": f"Failed after {max_attempts} attempts: {last_error}"}

        return wrapper
    return decorator


class ErrorLoopDetector:
    """
    Detects and prevents error loops in tool execution.

    Tracks tool execution errors and prevents repeated failures.
    """

    def __init__(self, max_errors: int = 2):
        """
        Args:
            max_errors: Maximum allowed consecutive errors for same tool+args
        """
        self.max_errors = max_errors
        self.error_history: dict[str, int] = {}

    def get_tool_key(self, tool_name: str, arguments: dict) -> str:
        """Generate unique key for tool+arguments"""
        import json
        args_str = json.dumps(arguments, sort_keys=True)
        return f"{tool_name}:{args_str}"

    def should_skip(self, tool_name: str, arguments: dict) -> tuple[bool, Optional[str]]:
        """
        Check if tool should be skipped due to repeated failures.

        Returns:
            (should_skip, reason)
        """
        key = self.get_tool_key(tool_name, arguments)
        error_count = self.error_history.get(key, 0)

        if error_count >= self.max_errors:
            reason = f"Tool {tool_name} has failed {error_count} times with same arguments, skipping"
            return True, reason

        return False, None

    def record_error(self, tool_name: str, arguments: dict):
        """Record a tool execution error"""
        key = self.get_tool_key(tool_name, arguments)
        self.error_history[key] = self.error_history.get(key, 0) + 1

    def record_success(self, tool_name: str, arguments: dict):
        """Record a tool execution success (clears error count)"""
        key = self.get_tool_key(tool_name, arguments)
        self.error_history.pop(key, None)

    def reset(self):
        """Reset all error history"""
        self.error_history.clear()


class FallbackStrategy:
    """
    Provides fallback strategies for failed operations.
    """

    @staticmethod
    def llm_timeout_fallback() -> str:
        """Fallback message for LLM timeout"""
        return """
抱歉，AI 响应超时。这可能是因为：
1. 请求过于复杂
2. API 服务繁忙
3. 网络连接问题

建议：
- 简化您的问题
- 稍后重试
- 或使用 /status 检查系统状态
"""

    @staticmethod
    def llm_error_fallback(error: Exception) -> str:
        """Fallback message for LLM error"""
        return f"""
抱歉，AI 服务遇到问题：{str(error)}

建议：
- 检查 API 密钥配置
- 检查网络连接
- 稍后重试
"""

    @staticmethod
    def tool_timeout_fallback(tool_name: str, timeout: int) -> dict:
        """Fallback result for tool timeout"""
        return {
            "error": f"工具 {tool_name} 执行超时 ({timeout}s)",
            "timeout": True,
            "suggestion": "尝试减小操作范围或增加超时时间"
        }

    @staticmethod
    def tool_error_fallback(tool_name: str, error: Exception) -> dict:
        """Fallback result for tool error"""
        return {
            "error": f"工具 {tool_name} 执行失败: {str(error)}",
            "suggestion": "检查参数是否正确，或联系管理员"
        }


# Timeout configurations
DEFAULT_TIMEOUTS = {
    "llm_call": 60,
    "tool_read_file": 30,
    "tool_write_file": 30,
    "tool_search_code": 60,
    "tool_run_command": 30,
    "tool_rag_search": 30,
    "tool_rag_update": 300,
    "tool_rag_rebuild": 600,
}


# Retry configurations
DEFAULT_RETRIES = {
    "llm_call": 3,
    "tool_execution": 2,
}


def get_timeout(operation: str) -> int:
    """Get timeout for an operation"""
    return DEFAULT_TIMEOUTS.get(operation, 30)


def get_retry_count(operation: str) -> int:
    """Get retry count for an operation"""
    return DEFAULT_RETRIES.get(operation, 2)

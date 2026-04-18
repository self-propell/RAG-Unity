"""
Enhanced tool execution with error handling

Wraps existing tools with timeout, retry, and error handling.
"""
from core.error_handling import (
    with_timeout,
    with_retry,
    ErrorLoopDetector,
    FallbackStrategy,
    get_timeout,
    get_retry_count,
    ToolTimeoutError,
    ToolRetryError
)


# Global error loop detector
_error_detector = ErrorLoopDetector(max_errors=2)


def execute_tool_safe(
    project_root: str,
    tool_name: str,
    arguments: dict,
    rag_context: dict = None,
) -> dict:
    """
    Safe tool execution with timeout, retry, and loop detection.

    This is a wrapper around the original execute_tool that adds:
    - Error loop detection
    - Retry logic
    - Timeout handling
    - Fallback strategies

    Args:
        project_root: Project root directory
        tool_name: Tool name
        arguments: Tool arguments
        rag_context: RAG context

    Returns:
        Tool execution result (always returns dict, never raises)
    """
    # Check if we should skip due to repeated failures
    should_skip, skip_reason = _error_detector.should_skip(tool_name, arguments)
    if should_skip:
        return {
            "error": skip_reason,
            "skipped": True,
            "suggestion": "尝试修改参数或使用其他工具"
        }

    # Import original execute_tool
    from core.tools import execute_tool

    # Get retry count for this tool
    max_retries = get_retry_count("tool_execution")

    # Execute with retry
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = execute_tool(
                project_root=project_root,
                tool_name=tool_name,
                arguments=arguments,
                rag_context=rag_context
            )

            # Check if result indicates error
            if isinstance(result, dict) and result.get("error"):
                # Record error
                _error_detector.record_error(tool_name, arguments)

                # If timeout and we have retries left, retry
                if result.get("timeout") and attempt < max_retries:
                    import time
                    time.sleep(1)
                    continue

                # Otherwise return the error result
                return result

            # Success - clear error history
            _error_detector.record_success(tool_name, arguments)
            return result

        except ToolTimeoutError as e:
            last_error = e
            _error_detector.record_error(tool_name, arguments)

            if attempt < max_retries:
                import time
                time.sleep(1)
                continue

            return FallbackStrategy.tool_timeout_fallback(
                tool_name,
                get_timeout(f"tool_{tool_name}")
            )

        except Exception as e:
            last_error = e
            _error_detector.record_error(tool_name, arguments)

            if attempt < max_retries:
                import time
                time.sleep(1)
                continue

            return FallbackStrategy.tool_error_fallback(tool_name, e)

    # All retries exhausted
    return {
        "error": f"工具执行失败（已重试 {max_retries} 次）: {last_error}",
        "retries_exhausted": True
    }


def reset_error_detector():
    """Reset the global error detector (useful for testing or new conversations)"""
    _error_detector.reset()


def get_error_history() -> dict:
    """Get current error history (for debugging)"""
    return _error_detector.error_history.copy()

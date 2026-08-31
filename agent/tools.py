from typing import Any


def build_tool_definition(tool):
    """
    Convert an MCP tool definition into a Gemini-compatible
    function declaration.
    """

    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }
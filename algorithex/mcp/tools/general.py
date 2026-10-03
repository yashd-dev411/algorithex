"""
Algorithex General Tools

This module provides MCP tools for general Algorithex operations.

The tools include:
- get_algorithex_status: Get the current status of Algorithex
- greet_user: Generate a greeting message for the user
"""

from algorithex.mcp.tools.services.general import (
    get_algorithex_status as get_algorithex_status_service,
    greet_user as greet_user_service,
)


def register_general_tools(mcp):
    """
    Register the tools for the general operations.

    Args:
        mcp: The MCP server instance.

    Returns:
        None
    """
    # Tool: get algorithex status
    @mcp.tool()
    def get_algorithex_status():
        return get_algorithex_status_service()

    # Tool: greet user
    @mcp.tool()
    def greet_user(name: str):
        return greet_user_service(name)


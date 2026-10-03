"""
Algorithex MCP (Model Context Protocol) Module

This module provides the MCP server functionality for Algorithex, allowing MCP clients
(like Cursor) to interact with the Algorithex trading framework.

Main exports:
- run_mcp_server: Start the MCP server
- terminate_mcp_server: Stop the MCP server

The MCP server runs in a separate subprocess and provides tools for interacting
with Algorithex's API and functionality.
"""

from algorithex.mcp.manager import run_mcp_server, terminate_mcp_server
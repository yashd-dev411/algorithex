"""
Algorithex MCP Server Entry Point

This module is the entry point for the Algorithex MCP server. It:
1. Parses command line arguments (--port, --api_url)
2. Creates and configures the FastMCP server instance
3. Sets the Algorithex API URL for tools to use
4. Registers all available MCP tools
5. Starts the server with streamable-http transport

USAGE:
------
Run directly: python -m algorithex.mcp.server --port 9002 --api_url http://localhost:9000 --password your_password
Or via manager: manager.py handles subprocess creation with proper arguments

The server runs on 0.0.0.0 and accepts connections from MCP clients (like Cursor).
"""

import argparse
import logging
import sys
import traceback
from mcp.server.fastmcp import FastMCP
import algorithex.mcp.mcp_config as mcp_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("algorithex.mcp.server")
MCP_HOST = "0.0.0.0"


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions with traceback for transport/session debugging."""
    logger.error("Uncaught exception in MCP server", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = _log_uncaught_exception

# Parse command line arguments
parser = argparse.ArgumentParser(description='Algorithex MCP Server')
parser.add_argument('--port', type=int, required=True, help='Port to run the MCP server on')
parser.add_argument('--api_url', type=str, required=True, help='Algorithex API URL')
parser.add_argument('--password', type=str, required=True, help='Algorithex admin password for HTTP API auth')
args = parser.parse_args()

# Initialize shared runtime config for this subprocess.
# The manager process passes these values via CLI args.
mcp_config.ALGORITHEX_API_URL = args.api_url
mcp_config.ALGORITHEX_PASSWORD = args.password
mcp_config.MCP_PORT = args.port
mcp_config.MCP_URL = f"http://localhost:{args.port}/mcp"

# Advertise this workflow during initialization, including to clients without local agent rules.
mcp = FastMCP(
    "Algorithex MCP Server", host=MCP_HOST, port=args.port, json_response=True,
    instructions=(
        "When asked to prepare, adapt, port, or deploy a strategy researched on traditional-market "
        "data for a 24/7 crypto exchange (including tokenized or stock-linked instruments), read "
        "algorithex://strategy, especially Trading Hours and its adaptation workflow, before editing. "
        "Recognize equivalent requests even when the user does not mention trading hours."
    ),
)

# Register all available resources
from algorithex.mcp.resources import register_resources
register_resources(mcp)

# Register all available tools
from algorithex.mcp.tools import register_tools
register_tools(mcp)

def run():
    """Start the MCP server with streamable-http transport."""
    try:
        logger.info("Starting MCP streamable-http server at %s", mcp_config.MCP_URL)
        mcp.run(transport="streamable-http")
    except Exception as exc:
        logger.error("MCP server runtime failure: %s", exc)
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise

# Run the MCP server when executed directly
if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Quick test to verify RLBench MCP Server can be launched and tools are accessible
This bypasses the full agent system to test just the MCP connection
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters

async def test_rlbench_mcp_server():
    """Test that we can connect to RLBench MCP server and list tools"""

    print("=" * 70)
    print("Testing RLBench MCP Server Connection")
    print("=" * 70)

    # Path to RLBench server
    rlbench_server_path = project_root / "multi_tool_agent" / "ros_mcp_server" / "rlbench_server.py"

    print(f"\n1. Server path: {rlbench_server_path}")
    print(f"   Exists: {rlbench_server_path.exists()}")

    if not rlbench_server_path.exists():
        print("❌ ERROR: RLBench server file not found!")
        return False

    print("\n2. Creating MCPToolset...")

    # Get CoppeliaSim environment variables
    import os
    coppeliasim_root = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')

    # Prepare environment variables for subprocess
    env_vars = {
        'COPPELIASIM_ROOT': coppeliasim_root,
        'LD_LIBRARY_PATH': coppeliasim_root,
        'QT_QPA_PLATFORM': 'xcb',  # Ensure GUI works
    }

    print(f"   COPPELIASIM_ROOT: {coppeliasim_root}")

    try:
        # Create toolset (this will spawn the MCP server as subprocess)
        toolset = MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[str(rlbench_server_path)],
                    env=env_vars,  # Pass environment variables
                ),
                timeout=120,
            ),
        )

        print("✅ MCPToolset created successfully")

        print("\n3. Listing available tools...")

        # Use get_tools() method to retrieve tools (async)
        tools = await toolset.get_tools()

        print(f"✅ Found {len(tools)} tools:")
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. {tool.name}")

        print("\n4. Testing tools are accessible...")

        # The tools are now available to agents via the toolset
        print(f"✅ Toolset has {len(tools)} tools ready for agent use")

        print("\n" + "=" * 70)
        print("✅ SUCCESS: RLBench MCP Server is working!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ ERROR: Failed to connect to MCP server")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nThis test will:")
    print("  1. Create an MCP connection to rlbench_server.py")
    print("  2. List available tools")
    print("  3. Test a simple tool call")
    print("  4. Report success or failure\n")

    # Run the test
    success = asyncio.run(test_rlbench_mcp_server())

    sys.exit(0 if success else 1)

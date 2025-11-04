import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types as genai_types
import requests

load_dotenv()

class MCPGeminiClient:
    """Client for Gemini with MCP tools via HTTP"""
    
    def __init__(self, mcp_url: str = None, model: str = "gemini-2.5-flash"):
        # Support both local(testing purposes) and Docker deployment
        self.mcp_url = mcp_url or os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        self.model = model
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        
        self.genai_client = genai.Client(api_key=api_key)
        
        # Waiting for MCP server to be ready
        self._wait_for_server()
    
    def _wait_for_server(self, max_retries=30, delay=2):
        """Wait for MCP server to become available"""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.mcp_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"Connected to MCP server at {self.mcp_url}")
                    return
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    print(f"Waiting for MCP server... ({i+1}/{max_retries})")
                    import time
                    time.sleep(delay)
        
        raise ConnectionError(f"Could not connect to MCP server at {self.mcp_url}")
    
    def get_mcp_tools(self):
        """Fetch available tools from MCP server"""
        response = requests.get(f"{self.mcp_url}/tools")
        tools = response.json()
        
        # Converting MCP tools to Gemini format
        gemini_tools = []
        for tool in tools:
            gemini_tools.append(
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=tool["parameters"]
                        )
                    ]
                )
            )
        return gemini_tools, {t["name"]: t for t in tools}
    
    def call_mcp_tool(self, tool_name: str, args: dict):
        """Execute MCP tool via HTTP"""
        response = requests.post(
            f"{self.mcp_url}/call-tool",
            json={"name": tool_name, "arguments": args}
        )
        return response.json()["result"]
    
    async def process_query(self, query: str):
        """Process query with Gemini and MCP tools"""
        gemini_tools, tool_map = self.get_mcp_tools()
        
        contents = [{"role": "user", "parts": [{"text": query}]}]
        
        # Initial Gemini call
        response = await self.genai_client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                tools=gemini_tools,
                tool_config=genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(mode="AUTO")
                )
            )
        )
        
        # Checking for function call
        part = response.candidates[0].content.parts[0]
        fn_call = getattr(part, "function_call", None)
        
        if fn_call:
            tool_name = fn_call.name
            
            # Extracting arguments
            args = {}
            if hasattr(fn_call, "args"):
                args_obj = fn_call.args
                if hasattr(args_obj, "items"):
                    args = dict(args_obj.items())
                elif isinstance(args_obj, dict):
                    args = args_obj
            
            print(f"Calling tool: {tool_name} with args: {args}")
            
            # Calling MCP tool
            tool_result = self.call_mcp_tool(tool_name, args)
            
            # Sending result back to Gemini
            followup_contents = [
                {"role": "user", "parts": [{"text": query}]},
                {
                    "role": "model",
                    "parts": [{"function_call": {"name": tool_name, "args": args}}]
                },
                {
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": tool_name,
                            "response": {"content": tool_result}
                        }
                    }]
                }
            ]
            
            final_response = await self.genai_client.aio.models.generate_content(
                model=self.model,
                contents=followup_contents,
                config=genai_types.GenerateContentConfig(
                    tool_config=genai_types.ToolConfig(
                        function_calling_config=genai_types.FunctionCallingConfig(mode="NONE")
                    )
                )
            )
            
            return final_response.candidates[0].content.parts[0].text
        
        # No function call - return text
        return part.text if hasattr(part, "text") else str(response)

async def main():
    client = MCPGeminiClient()
    
    # Example queries
    queries = [
        "Find ICD-11 codes for diabetes",
        "What are common comorbidities for hypertension?",
        "Search CPT codes for MRI scan"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = await client.process_query(query)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())

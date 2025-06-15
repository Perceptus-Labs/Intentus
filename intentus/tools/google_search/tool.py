import os
import aiohttp
from typing import List, Dict, Any
from dataclasses import dataclass

from intentus.tools.base import BaseTool

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Google_Search_Tool(BaseTool):
    """Tool for performing Google searches."""

    name: str = "Google_Search_Tool"
    description: str = "A tool for performing Google searches and retrieving results"
    version: str = "1.0.0"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cx = os.getenv("GOOGLE_CX")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_types": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                    "required": True,
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "required": False,
                    "default": 5,
                },
            },
            "output_types": {
                "results": {
                    "type": "array",
                    "description": "List of search results",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "link": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                }
            },
        }

    async def execute(self, command: str) -> Dict[str, Any]:
        """Execute the Google search."""
        if not self.api_key:
            return {
                "success": False,
                "error": "Google API key is not set. Please set the GOOGLE_API_KEY environment variable.",
                "result": None,
            }

        try:
            # Parse the command to get query and num_results
            # For now, just use the command as the query
            query = command
            num_results = 5

            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "key": self.api_key,
                    "cx": self.cx,
                    "num": num_results,
                }
                async with session.get(self.base_url, params=params) as response:
                    results = await response.json()

                    if "items" in results:
                        return {
                            "success": True,
                            "error": None,
                            "result": {
                                "results": [
                                    {
                                        "title": item["title"],
                                        "link": item["link"],
                                        "snippet": item["snippet"],
                                    }
                                    for item in results["items"]
                                ]
                            },
                        }
                    else:
                        return {
                            "success": False,
                            "error": "No results found.",
                            "result": None,
                        }
        except Exception as e:
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}",
                "result": None,
            }


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    export GOOGLE_API_KEY=your_api_key_here
    cd octotools/tools/google_search
    python tool.py
    """

    # Example usage of the Google_Search_Tool
    tool = Google_Search_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Execute the tool to perform a Google search
    query = "nobel prize winners in chemistry 2024"
    try:
        execution = tool.execute(query=query)
        print("\nExecution Result:")
        print(f"Search query: {query}")
        print(f"Number of results: {len(execution['result']['results'])}")
        print("\nSearch Results:")
        if execution["error"]:
            print(f"Error: {execution['error']}")
        else:
            for i, item in enumerate(execution["result"]["results"], 1):
                print(f"\n{i}. Title: {item['title']}")
                print(f"   URL: {item['link']}")
                print(f"   Snippet: {item['snippet']}")
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")

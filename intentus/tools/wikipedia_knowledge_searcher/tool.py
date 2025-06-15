import os
import wikipedia
from ..base import BaseTool
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class Wikipedia_Knowledge_Searcher_Tool(BaseTool):
    """
    A tool that searches Wikipedia using keywords or search terms and returns relevant article content.
    The input should be a simple keyword or search term, not a full sentence or question.
    """

    def __init__(self):
        super().__init__(
            tool_name="Wikipedia_Knowledge_Searcher_Tool",
            tool_description="A tool that searches Wikipedia using keywords or search terms and returns relevant article content. The input should be a simple keyword or search term, not a full sentence or question.",
            tool_version="1.0.0",
            input_types={
                "query": {
                    "type": "string",
                    "description": "A simple keyword or search term to look up on Wikipedia (e.g., 'Paris', 'Quantum Physics', 'French Revolution'). Do not use full sentences or questions.",
                    "required": True,
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of the returned text (default: 2000)",
                    "required": False,
                    "default": 2000,
                },
            },
            output_type={
                "results": {
                    "type": "object",
                    "description": "Wikipedia search results and content",
                    "properties": {
                        "search_results": {
                            "type": "array",
                            "description": "List of search results",
                            "items": {"type": "string"},
                        },
                        "content": {
                            "type": "string",
                            "description": "Extracted Wikipedia content",
                        },
                    },
                }
            },
        )
        self.max_length = 2000
        wikipedia.set_lang("en")

    def search_wikipedia(self, query: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        Search Wikipedia for a given query and return relevant content.

        Args:
            query (str): The search query
            max_length (int): Maximum length of the returned text

        Returns:
            Dict[str, Any]: A dictionary containing search results and content
        """
        self.logger.debug(f"Searching Wikipedia with query: {query}")
        try:
            # Search for the query
            search_results = wikipedia.search(query)
            self.logger.debug(f"Search results: {search_results}")

            if not search_results:
                return {
                    "search_results": [],
                    "content": f"No results found for query: {query}",
                }

            # Get the content of the first result
            try:
                page = wikipedia.page(search_results[0])
                content = page.content[:max_length]
                return {
                    "search_results": search_results,
                    "content": content,
                }
            except wikipedia.exceptions.DisambiguationError as e:
                # If the page is a disambiguation page, return the options
                return {
                    "search_results": search_results,
                    "content": f"Disambiguation page. Options: {', '.join(e.options)}",
                }
            except wikipedia.exceptions.PageError:
                return {
                    "search_results": search_results,
                    "content": f"Page not found for: {search_results[0]}",
                }

        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return {
                "search_results": [],
                "content": f"Error searching Wikipedia: {str(e)}",
            }

    async def execute(self, command: str) -> Dict[str, Any]:
        """
        Execute the Wikipedia search tool.

        Args:
            command (str): The search query or command to execute

        Returns:
            Dict[str, Any]: The search results and content
        """
        self.logger.debug(f"Executing Wikipedia search with command: {command}")

        try:
            # Extract the search query from the command
            # The command should be a simple keyword or search term
            search_query = command.strip()

            # Perform the search
            results = self.search_wikipedia(search_query)
            self.logger.debug(
                f"Search completed. Found {len(results['search_results'])} results"
            )
            self.logger.debug(f"Content length: {len(results['content'])} characters")

            return results

        except Exception as e:
            self.logger.error(f"Error executing Wikipedia search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        tool = Wikipedia_Knowledge_Searcher_Tool()

        # Get tool metadata
        metadata = tool.get_metadata()
        print("Tool Metadata:")
        print(metadata)

        # Test queries
        test_queries = [
            "Python programming language",
            "Artificial Intelligence",
            "Theory of Relativity",
        ]

        for query in test_queries:
            print(f"\nTesting query: {query}")
            result = await tool.execute(query)
            print("Result:")
            print(result)

    asyncio.run(main())

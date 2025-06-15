import os
import wikipedia
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from intentus.tools.base import BaseTool

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Wikipedia_Knowledge_Searcher_Tool(BaseTool):
    """Tool for searching Wikipedia and retrieving article content."""

    name: str = "Wikipedia_Knowledge_Searcher_Tool"
    description: str = (
        "A tool that searches Wikipedia and returns web text based on a given query."
    )
    version: str = "1.0.0"

    def __init__(self):
        super().__init__()
        self.max_length = 2000
        # Set language to English
        wikipedia.set_lang("en")

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_types": {
                "query": {
                    "type": "string",
                    "description": "The search query for Wikipedia",
                    "required": True,
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of the returned text (default: 2000)",
                    "required": False,
                    "default": 2000,
                },
            },
            "output_types": {
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
        }

    def search_wikipedia(
        self, query: str, max_length: int = 2000
    ) -> Tuple[List[str], str]:
        """
        Searches Wikipedia based on the given query and returns the text.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): The maximum length of the returned text. Use -1 for full text.

        Returns:
            tuple: (search_results, page_text)
        """
        logger.debug(f"Searching Wikipedia with query: {query}")
        try:
            # First, search for matching pages
            search_results = wikipedia.search(query, results=5)
            logger.debug(f"Search results: {search_results}")

            if not search_results:
                return [], "No results found for the given query."

            # Get the first result's page
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                text = page.content

                if max_length != -1:
                    text = text[:max_length]

                return search_results, text
            except wikipedia.exceptions.DisambiguationError as e:
                # If we get a disambiguation page, try the first option
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    text = page.content
                    if max_length != -1:
                        text = text[:max_length]
                    return e.options, text
                except Exception as e2:
                    return e.options, f"Error accessing disambiguation page: {str(e2)}"
            except wikipedia.exceptions.PageError:
                return (
                    search_results,
                    f"PageError: No Wikipedia page found for '{query}'.",
                )
            except Exception as e:
                return search_results, f"Error accessing page: {str(e)}"
        except Exception as e:
            logger.error(f"Error in search_wikipedia: {str(e)}")
            return [], f"Error searching Wikipedia: {str(e)}"

    async def execute(self, command: str) -> Dict[str, Any]:
        """
        Searches Wikipedia based on the provided query and returns the results.

        Parameters:
            command (str): The search query for Wikipedia.

        Returns:
            dict: A dictionary containing the search results and extracted text.
        """
        logger.debug(f"Executing Wikipedia search with command: {command}")
        try:
            search_results, text = self.search_wikipedia(command, self.max_length)
            logger.debug(f"Search results: {search_results}")
            logger.debug(f"Text length: {len(text) if text else 0}")

            if not search_results:
                return {"success": False, "error": text, "result": None}

            return {
                "success": True,
                "error": None,
                "result": {"search_results": search_results, "content": text},
            }
        except Exception as e:
            logger.error(f"Error in execute: {str(e)}")
            return {
                "success": False,
                "error": f"Error executing Wikipedia search: {str(e)}",
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

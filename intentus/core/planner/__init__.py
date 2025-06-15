import os
import re
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging

from ..config import CoreConfig
from ..engine.factory import create_llm_engine
from ..memory import Memory
from ..formatters import QueryAnalysis, NextStep, MemoryVerification

# Set up logging
logger = logging.getLogger(__name__)


class Planner:
    """Planner class for Intentus agent."""

    def __init__(
        self,
        llm_engine_name: str,
        toolbox_metadata: Dict[str, Any],
        available_tools: List[str],
        verbose: bool = True,
    ):
        """Initialize the planner."""
        logger.debug(f"Initializing Planner with engine: {llm_engine_name}")
        logger.debug(f"Available tools: {available_tools}")
        logger.debug(f"Toolbox metadata: {json.dumps(toolbox_metadata, indent=2)}")

        self.llm_engine = create_llm_engine(llm_engine_name)
        self.toolbox_metadata = toolbox_metadata
        self.available_tools = available_tools
        self.verbose = verbose
        self.query_analysis = None
        self.context = None
        self.subgoal = None
        self.tool = None
        self.command = None
        self.final_output = None
        self.direct_output = None
        logger.debug("Planner initialized")

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image information."""
        logger.debug(f"Getting image info for: {image_path}")
        if not image_path:
            logger.debug("No image path provided")
            return {}
        logger.debug(f"Image path exists: {os.path.exists(image_path)}")
        return {"image_path": image_path}

    async def generate_base_response(self, question: str, image: str) -> str:
        """Generate base response."""
        logger.debug(f"Generating base response for question: {question}")
        logger.debug(f"Image provided: {image}")

        image_info = self.get_image_info(image)
        logger.debug(f"Image info: {image_info}")

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
                logger.debug("Successfully read image file")
            except Exception as e:
                logger.error(f"Error reading image file: {str(e)}")

        logger.debug("Calling LLM engine for base response")
        self.base_response = await self.llm_engine(question)
        logger.debug(f"Base response generated: {self.base_response}")

        return self.base_response

    async def analyze_query(self, question: str, image: str) -> str:
        """Analyze the query and determine required skills."""
        logger.debug(f"Analyzing query: {question}")
        logger.debug(f"Image provided: {image}")

        prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Image: {image}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Provide a brief explanation for each skill you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""
        logger.debug("Calling LLM engine for query analysis")
        response = await self.llm_engine(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "QueryAnalysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "concise_summary": {"type": "string"},
                            "required_skills": {"type": "string"},
                            "additional_considerations": {"type": "string"},
                        },
                        "required": [
                            "concise_summary",
                            "required_skills",
                            "additional_considerations",
                        ],
                    },
                },
            },
        )
        logger.debug(f"Query analysis response: {response}")

        self.query_analysis = response
        return str(response).strip()

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:
        """Extract context, subgoal, and tool from the response."""
        logger.debug(f"Extracting context, subgoal, and tool from response: {response}")
        logger.debug(f"Response type: {type(response)}")

        try:
            # If response is already a dict, use it directly
            if isinstance(response, dict):
                data = response
            else:
                # Try to parse as JSON
                import json

                data = json.loads(str(response))

            logger.debug(f"Parsed data: {data}")

            # Extract values from the parsed data
            context = data.get("context", "")
            subgoal = data.get("sub_goal", "")
            tool = data.get("tool_name", "")

            logger.debug(f"Final extracted values:")
            logger.debug(f"Context: '{context}'")
            logger.debug(f"Subgoal: '{subgoal}'")
            logger.debug(f"Tool: '{tool}'")

            return context, subgoal, tool

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            # Fallback to old string parsing method
            lines = str(response).split("\n")
            context = ""
            subgoal = ""
            tool = ""

            for line in lines:
                if line.startswith("Context:"):
                    context = line.replace("Context:", "").strip()
                elif line.startswith("Sub-Goal:"):
                    subgoal = line.replace("Sub-Goal:", "").strip()
                elif line.startswith("Tool:"):
                    tool = line.replace("Tool:", "").strip()
                elif line.startswith("Tool Name:"):
                    tool = line.replace("Tool Name:", "").strip()

            logger.debug(f"Fallback extracted values:")
            logger.debug(f"Context: '{context}'")
            logger.debug(f"Subgoal: '{subgoal}'")
            logger.debug(f"Tool: '{tool}'")

            return context, subgoal, tool

    async def generate_next_step(
        self,
        question: str,
        image: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
    ) -> Any:
        """Generate next step."""
        logger.debug(f"Generating next step for question: {question}")
        logger.debug(f"Current step: {step_count + 1} of {max_step_count}")
        logger.debug(f"Query analysis: {query_analysis}")

        prompt = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.available_tools}

Tool Metadata:
{self.toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
   - Key objectives from the query analysis
   - Capabilities of available tools
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format:
Your response MUST follow this structure:
1. Justification: Explain your choice in detail.
2. Context, Sub-Goal, and Tool: Present the context, sub-goal, and the selected tool ONCE with the following format:

Context: <context>
Sub-Goal: <sub_goal>
Tool Name: <tool_name>

Where:
- <context> MUST include ALL necessary information for the tool to function, structured as follows:
  * Relevant data from previous steps
  * File names or paths created or used in previous steps (list EACH ONE individually)
  * Variable names and their values from previous steps' results
  * Any other context-specific information required by the tool
- <sub_goal> is a specific, achievable objective for the tool, based on its metadata and previous outcomes.
It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
- <tool_name> MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {self.available_tools}.
"""

        logger.debug("Calling LLM engine for next step generation")
        response = await self.llm_engine(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "NextStep",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "justification": {"type": "string"},
                            "context": {"type": "string"},
                            "sub_goal": {"type": "string"},
                            "tool_name": {"type": "string"},
                        },
                        "required": [
                            "justification",
                            "context",
                            "sub_goal",
                            "tool_name",
                        ],
                    },
                },
            },
        )
        logger.debug(f"Raw LLM response for next step: {response}")
        logger.debug(f"Response type: {type(response)}")
        if isinstance(response, dict):
            logger.debug(f"Response keys: {response.keys()}")
            logger.debug(
                f"Tool name from response: {response.get('tool_name', 'NOT FOUND')}"
            )
        else:
            logger.debug(f"Response as string: {str(response)}")
            # Log the exact format of the response
            logger.debug("Response format analysis:")
            for line in str(response).split("\n"):
                logger.debug(f"Line: '{line}'")

        return response

    async def verificate_context(
        self, question: str, image: str, query_analysis: str, memory: Memory
    ) -> Any:
        """Verify context and determine if we should stop."""
        logger.debug("Verifying context and checking stop condition")
        logger.debug(f"Question: {question}")
        logger.debug(f"Query analysis: {query_analysis}")

        prompt = f"""
Task: Verify if the current context and results are sufficient to answer the query.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Previous Steps and Their Results:
{memory.get_actions()}

Instructions:
1. Review the query and its analysis.
2. Evaluate the results from previous steps.
3. Determine if we have enough information to answer the query.
4. Decide whether to continue or stop.

Response Format:
Your response MUST follow this structure:
1. Analysis: Explain your reasoning in detail.
2. Conclusion: Either "CONTINUE" or "STOP".

Rules:
- If we have enough information to answer the query, conclude with "STOP".
- If we need more information or steps, conclude with "CONTINUE".
- Be thorough in your analysis.
"""

        logger.debug("Calling LLM engine for context verification")
        response = await self.llm_engine(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "MemoryVerification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis": {"type": "string"},
                            "stop_signal": {"type": "string"},
                        },
                        "required": ["analysis", "stop_signal"],
                    },
                },
            },
        )
        logger.debug(f"Verification response: {response}")

        return response

    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        """Extract analysis and conclusion from verification response."""
        logger.debug(f"Extracting conclusion from response: {response}")

        # Parse the response to extract analysis and conclusion
        lines = str(response).split("\n")
        analysis = ""
        conclusion = ""

        for line in lines:
            if line.startswith("Analysis:"):
                analysis = line.replace("Analysis:", "").strip()
            elif line.startswith("Conclusion:"):
                conclusion = line.replace("Conclusion:", "").strip()

        logger.debug(f"Extracted analysis: {analysis}")
        logger.debug(f"Extracted conclusion: {conclusion}")

        return analysis, conclusion

    async def generate_final_output(
        self, question: str, image: str, memory: Memory
    ) -> str:
        """Generate final output."""
        logger.debug("Generating final output")
        logger.debug(f"Question: {question}")

        prompt = f"""
Task: Generate a comprehensive final answer to the query based on all previous steps and their results.

Context:
Query: {question}
Image: {image}

Previous Steps and Their Results:
{memory.get_actions()}

Instructions:
1. Review all previous steps and their results.
2. Synthesize the information into a coherent answer.
3. Ensure the answer directly addresses the query.
4. Present the information in a clear, structured format.

Response Format:
Your response should be a well-structured answer that:
1. Directly addresses the query
2. Incorporates relevant information from all steps
3. Is clear and easy to understand
4. Provides a complete and accurate response
"""

        logger.debug("Calling LLM engine for final output generation")
        response = await self.llm_engine(prompt)
        logger.debug(f"Final output generated: {response}")

        self.final_output = response
        return response

    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        """Generate a direct output without using tools."""
        image_info = self.get_image_info(image)

        prompt_direct_output = f"""
Task: Generate a direct response to the query without using any tools.

Query: {question}
Image: {image_info}

Previous Steps and Their Results:
{memory.get_actions()}

Instructions:
1. Review the original query and any accompanying inputs.
2. Generate a direct response based on your knowledge.
3. Ensure the response is clear and directly addresses the query.
4. Include relevant details and explanations where necessary.

Your response should:
1. Be clear and concise
2. Directly answer the query
3. Be well-structured and easy to understand

Please provide your response in a clear, structured format.
"""

        input_data = [prompt_direct_output]
        if image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        final_output = self.llm_engine(input_data)
        return final_output

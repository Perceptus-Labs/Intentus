import os
import re
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

from ..config import CoreConfig
from ..engine.factory import create_llm_engine
from ..memory import Memory
from ..formatters import QueryAnalysis, NextStep, MemoryVerification


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

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image information."""
        if not image_path:
            return {}
        return {"image_path": image_path}

    async def generate_base_response(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = await self.llm_engine(question)

        return self.base_response

    async def analyze_query(self, question: str, image: str) -> str:
        """Analyze the query and determine required skills."""
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

        self.query_analysis = response
        return str(response).strip()

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:
        """Extract context, subgoal, and tool from the response."""
        # Parse the response to extract context, subgoal, and tool
        # This is a simplified version - you might want to make it more robust
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

        return response

    async def verificate_context(
        self, question: str, image: str, query_analysis: str, memory: Memory
    ) -> Any:
        """Verify if the context is complete."""
        prompt_verificate_context = f"""
Task: Verify if the current context is complete and if the query has been answered.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Previous Steps and Their Results:
{memory.get_actions()}

Instructions:
1. Analyze the current context and previous steps.
2. Determine if the query has been answered.
3. If not, identify what is missing.

Response Format:
Your response MUST follow this structure:
1. Analysis: Explain your reasoning.
2. Conclusion: Either "CONTINUE" or "STOP".

Rules:
- "CONTINUE" if more steps are needed.
- "STOP" if the query has been answered.
"""

        response = await self.llm_engine(
            prompt_verificate_context,
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

        return response

    def extract_conclusion(self, response: Any) -> Tuple[str, str]:
        """Extract conclusion from the response."""
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            conclusion = response.stop_signal
        else:
            # Parse the response to extract analysis and conclusion
            lines = str(response).split("\n")
            analysis = ""
            conclusion = ""

            for line in lines:
                if line.startswith("Analysis:"):
                    analysis = line.replace("Analysis:", "").strip()
                elif line.startswith("Conclusion:"):
                    conclusion = line.replace("Conclusion:", "").strip()

        return analysis, conclusion

    async def generate_final_output(
        self, question: str, image: str, memory: Memory
    ) -> str:
        """Generate the final output based on the task execution results."""
        image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Task: Generate a comprehensive final output based on the task execution results.

Query: {question}
Image: {image_info}

Previous Steps and Their Results:
{memory.get_actions()}

Instructions:
1. Review the original query and any accompanying inputs.
2. Analyze the results from all previous steps.
3. Synthesize the information into a clear, coherent response.
4. Ensure the response directly addresses the original query.
5. Include relevant details and explanations where necessary.

Your response should:
1. Be clear and concise
2. Directly answer the query
3. Include relevant details from the execution results
4. Be well-structured and easy to understand

Please provide your final output in a clear, structured format.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        final_output = await self.llm_engine(input_data)
        return str(final_output).strip()

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

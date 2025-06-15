import os
import re
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json

from ..config import PlannerConfig
from ..engine.factory import create_llm_engine
from ..memory import Memory
from ..formatters import QueryAnalysis, NextStep, MemoryVerification


class Planner:
    """Planner class for Intentus agent."""

    def __init__(self, llm_engine: Any, config: PlannerConfig):
        """Initialize the planner."""
        self.llm_engine = llm_engine
        self.config = config
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

    def generate_base_response(
        self, question: str, image: str, max_tokens: str = 4000
    ) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)

        return self.base_response

    async def analyze_query(self, question: str, image: str) -> str:
        """Analyze the query and determine required skills."""
        input_data = [
            {
                "role": "user",
                "content": f"""
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
""",
            }
        ]

        response = await self.llm_engine(
            input_data,
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

    def generate_next_step(
        self,
        question: str,
        image: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
    ) -> Any:
        prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image}
Query Analysis: {query_analysis}

Available Tools:
{self.toolbox.tools}

Tool Metadata:
{self.toolbox.metadata}

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
- The tool name MUST exactly match one from the available tools list: {self.toolbox.tools}.
- Avoid redundancy by considering previous steps and building on prior results.
- Your response MUST conclude with the Context, Sub-Goal, and Tool Name sections IN THIS ORDER, presented ONLY ONCE.
- Include NO content after these three sections.

Example (do not copy, use only as reference):
Justification: [Your detailed explanation here]
Context: Image path: "example/image.jpg", Previous detection results: [list of objects]
Sub-Goal: Detect and count the number of specific objects in the image "example/image.jpg"
Tool Name: Object_Detector_Tool

Remember: Your response MUST end with the Context, Sub-Goal, and Tool Name sections, with NO additional content afterwards.
"""
        next_step = self.llm_engine(prompt_generate_next_step, response_format=NextStep)
        return next_step

    def verificate_context(
        self, question: str, image: str, query_analysis: str, memory: Memory
    ) -> Any:
        image_info = self.get_image_info(image)

        prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.toolbox.tools}
Toolbox Metadata: {self.toolbox.metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions()}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.

Response Format:

If the memory is complete, accurate, AND verified:
Explanation: 
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation: 
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion.
"""

        input_data = [prompt_memory_verification]
        if image_info:
            try:
                with open(image_info["image_path"], "rb") as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        stop_verification = self.llm_engine(
            input_data, response_format=MemoryVerification
        )

        return stop_verification

    def extract_conclusion(self, response: Any) -> str:
        if isinstance(response, MemoryVerification):
            analysis = response.analysis
            stop_signal = response.stop_signal
            if stop_signal:
                return analysis, "STOP"
            else:
                return analysis, "CONTINUE"
        else:
            analysis = response
            pattern = r"conclusion\**:?\s*\**\s*(\w+)"
            matches = list(re.finditer(pattern, response, re.IGNORECASE | re.DOTALL))
            # if match:
            #     conclusion = match.group(1).upper()
            #     if conclusion in ['STOP', 'CONTINUE']:
            #         return conclusion
            if matches:
                conclusion = matches[-1].group(1).upper()
                if conclusion in ["STOP", "CONTINUE"]:
                    return analysis, conclusion

            # If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
            if "stop" in response.lower():
                return analysis, "STOP"
            elif "continue" in response.lower():
                return analysis, "CONTINUE"
            else:
                print(
                    "No valid conclusion (STOP or CONTINUE) found in the response. Continuing..."
                )
                return analysis, "CONTINUE"

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

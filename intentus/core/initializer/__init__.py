import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import time
from intentus.tools.base import BaseTool


class Initializer:
    def __init__(
        self,
        enabled_tools: List[str] = [],
        llm_engine: str = None,
        verbose: bool = False,
        config_path: str = None,
    ):
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.load_all = self.enabled_tools == ["all"]
        self.llm_engine = llm_engine  # llm engine name
        self.verbose = verbose
        self.vllm_server_process = None
        self.config_path = config_path
        print("\n==> Initializing core...")
        print(f"Enabled tools: {self.enabled_tools}")
        print(f"LLM engine name: {self.llm_engine}")
        self._set_up_tools()

        # if vllm, set up the vllm server
        if llm_engine and llm_engine.startswith("vllm-"):
            self.setup_vllm_server()

    def get_project_root(self) -> str:
        """Get the project root directory."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate up to the intentus directory
        while os.path.basename(
            current_dir
        ) != "intentus" and current_dir != os.path.dirname(current_dir):
            current_dir = os.path.dirname(current_dir)

        if os.path.basename(current_dir) != "intentus":
            raise ValueError("Could not find intentus directory")

        return current_dir

    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        """Load tools and get their metadata."""
        print("Loading tools and getting metadata...")

        # Get project root and tools directory
        project_root = self.get_project_root()
        tools_dir = os.path.join(project_root, "tools")

        print(f"Project root: {project_root}")
        print(f"Tools directory: {tools_dir}")

        # Add project root to Python path
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Updated Python path: {sys.path}")

        # Scan tools directory
        tools_metadata = {}
        for tool_dir in os.listdir(tools_dir):
            tool_path = os.path.join(tools_dir, tool_dir)
            if os.path.isdir(tool_path) and not tool_dir.startswith("__"):
                tool_file = os.path.join(tool_path, "tool.py")
                if os.path.exists(tool_file):
                    print(f"\n==> Attempting to import: intentus.tools.{tool_dir}.tool")
                    try:
                        module = importlib.import_module(
                            f"intentus.tools.{tool_dir}.tool"
                        )
                        for item_name in dir(module):
                            item = getattr(module, item_name)
                            if (
                                isinstance(item, type)
                                and issubclass(item, BaseTool)
                                and item != BaseTool
                            ):
                                # Only load enabled tools or all tools if load_all is True
                                if self.load_all or item_name in self.enabled_tools:
                                    print(f"Found tool class: {item_name}")
                                    tool_instance = item()
                                    metadata = tool_instance.get_metadata()
                                    tools_metadata[item_name] = metadata
                                    print(f"Metadata for {item_name}: {metadata}")
                    except Exception as e:
                        print(f"Error importing {tool_dir}: {str(e)}")
                        print("Full traceback:")
                        print(traceback.format_exc())

        print(f"\n==> Total number of tools imported: {len(tools_metadata)}")
        return tools_metadata

    def run_demo_commands(self) -> List[str]:
        print("\n==> Running demo commands for each tool...")
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"Checking availability of {tool_name}...")

            try:
                # Get the tool directory name from the tool name
                tool_dir = tool_name.lower().replace("_tool", "")

                # Import the tool module
                module_name = f"intentus.tools.{tool_dir}.tool"
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, tool_name)

                # Instantiate the tool
                tool_instance = tool_class()

                # Add to available tools
                self.available_tools.append(tool_name)
                print(f"Successfully initialized {tool_name}")

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print("Full traceback:")
                print(traceback.format_exc())

        print("\n✅ Finished running demo commands for each tool.")
        return self.available_tools

    def _set_up_tools(self) -> None:
        print("\n==> Setting up tools...")

        # Load tools and get metadata
        self.toolbox_metadata = self.load_tools_and_get_metadata()

        # Run demo commands to determine available tools
        self.run_demo_commands()

        print("✅ Finished setting up tools.")
        print(f"✅ Total number of final available tools: {len(self.available_tools)}")
        print(f"✅ Final available tools: {self.available_tools}")

    def setup_vllm_server(self) -> None:
        # Check if vllm is installed
        try:
            import vllm
        except ImportError:
            raise ImportError(
                "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm`."
            )

        # Start the VLLM server
        command = [
            "vllm",
            "serve",
            self.llm_engine.replace("vllm-", ""),
            "--port",
            "8888",
        ]
        if self.config_path is not None:
            command = [
                "vllm",
                "serve",
                "--config",
                self.config_path,
                "--port",
                "8888",
            ]

        import subprocess

        vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        print("Starting VLLM server...")
        while True:
            output = vllm_process.stdout.readline()
            error = vllm_process.stderr.readline()
            time.sleep(5)
            if output.strip() != "":
                print("VLLM server standard output:", output.strip())
            if error.strip() != "":
                print("VLLM server error output:", error.strip())
            if "Server started" in output:
                print("VLLM server started successfully!")
                break


if __name__ == "__main__":
    enabled_tools = ["Google_Search_Tool"]
    initializer = Initializer(enabled_tools=enabled_tools)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)

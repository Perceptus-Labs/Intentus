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
        vllm_config_path: str = None,
    ):
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.load_all = self.enabled_tools == ["all"]
        self.llm_engine = llm_engine  # llm engine name
        self.verbose = verbose
        self.vllm_server_process = None
        self.vllm_config_path = vllm_config_path
        print("\n==> Initializing core...")
        print(f"Enabled tools: {self.enabled_tools}")
        print(f"LLM engine name: {self.llm_engine}")
        self._set_up_tools()

        # if vllm, set up the vllm server
        if llm_engine.startswith("vllm-"):
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
            if os.path.isdir(tool_path):
                tool_file = os.path.join(tool_path, "tool.py")
                if os.path.exists(tool_file):
                    print(f"\n==> Attempting to import: tools.{tool_dir}.tool")
                    try:
                        module = importlib.import_module(f"tools.{tool_dir}.tool")
                        for item_name in dir(module):
                            item = getattr(module, item_name)
                            if (
                                isinstance(item, type)
                                and issubclass(item, BaseTool)
                                and item != BaseTool
                            ):
                                print(f"Found tool class: {item_name}")
                                tool_instance = item()
                                metadata = tool_instance.get_metadata()
                                tools_metadata[item_name] = metadata
                                print(f"Metadata for {item_name}: {metadata}")
                    except Exception as e:
                        print(f"Error importing {tool_dir}: {str(e)}")

        print(f"\n==> Total number of tools imported: {len(tools_metadata)}")
        return tools_metadata

    def run_demo_commands(self) -> List[str]:
        print("\n==> Running demo commands for each tool...")
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"Checking availability of {tool_name}...")

            try:
                # Import the tool module
                module_name = f"tools.{tool_name.lower().replace('_tool', '')}.tool"
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, tool_name)

                # Instantiate the tool
                tool_instance = tool_class()

                # FIXME This is a temporary workaround to avoid running demo commands
                self.available_tools.append(tool_name)

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print(traceback.format_exc())

        # update the toolmetadata with the available tools
        self.toolbox_metadata = {
            tool: self.toolbox_metadata[tool] for tool in self.available_tools
        }
        print("\n✅ Finished running demo commands for each tool.")
        # print(f"Updated total number of available tools: {len(self.toolbox_metadata)}")
        # print(f"Available tools: {self.available_tools}")
        return self.available_tools

    def _set_up_tools(self) -> None:
        print("\n==> Setting up tools...")

        # Keep enabled tools
        self.available_tools = [
            tool.lower().replace("_tool", "") for tool in self.enabled_tools
        ]

        # Load tools and get metadata
        self.load_tools_and_get_metadata()

        # Run demo commands to determine available tools
        self.run_demo_commands()

        # Filter toolbox_metadata to include only available tools
        self.toolbox_metadata = {
            tool: self.toolbox_metadata[tool] for tool in self.available_tools
        }
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
        if self.vllm_config_path is not None:
            command = [
                "vllm",
                "serve",
                "--config",
                self.vllm_config_path,
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
                print("VLLM server standard error:", error.strip())

            if (
                "Application startup complete." in output
                or "Application startup complete." in error
            ):
                print("VLLM server started successfully.")
                break

            if vllm_process.poll() is not None:
                print(
                    "VLLM server process terminated unexpectedly. Please check the output above for more information."
                )
                break

        self.vllm_server_process = vllm_process


if __name__ == "__main__":
    enabled_tools = ["Generalist_Solution_Generator_Tool"]
    initializer = Initializer(enabled_tools=enabled_tools)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)

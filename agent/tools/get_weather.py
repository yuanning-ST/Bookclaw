from langchain_core.tools import BaseTool


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the current weather for a specified location."


    def _run(self, location: str) -> str:
        """Use the tool.

        Add `run_manager: CallbackManagerForToolRun | None = None` to child
        implementations to enable tracing.

        Returns:
            The result of the tool execution.
        """
        # Placeholder implementation, replace with actual weather fetching logic
        return f"The current weather in {location} is sunny with a temperature of 25°C."

    
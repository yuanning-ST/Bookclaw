"""
Get Skill Tool - Retrieve skill information by name.
"""
from typing import Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


class GetSkillArgs(BaseModel):
    """Arguments for GetSkillTool."""
    skill_name: str = Field(description="Name of the skill to retrieve")


class GetSkillTool(BaseTool):
    """Tool to retrieve skill information by name."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "get_skill"
    description: str = "Get detailed information about a specific skill by name. Returns the skill's description, triggers, and instructions."
    args_schema: type[BaseModel] = GetSkillArgs

    # Will be set by AgentGraph during initialization
    context_manager: Any = None

    def _run(self, skill_name: str) -> str:
        """Run the tool synchronously."""
        return self._get_skill_info(skill_name)

    async def _arun(self, skill_name: str) -> str:
        """Run the tool asynchronously."""
        return self._get_skill_info(skill_name)

    def _get_skill_info(self, skill_name: str) -> str:
        """Get skill information."""
        if not self.context_manager:
            return "Error: Context manager not initialized"

        skill = self.context_manager.get_skill(skill_name)
        if not skill:
            available = list(self.context_manager.get_skills().keys())
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        lines = [
            f"# {skill.name}",
            "",
            f"**描述**: {skill.description}",
        ]

        if skill.triggers:
            lines.append(f"**触发条件**:")
            for trigger in skill.triggers:
                lines.append(f"- {trigger}")

        lines.append("")
        lines.append("## 指令")
        lines.append(skill.instructions)

        return "\n".join(lines)
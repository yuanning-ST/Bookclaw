"""
Context Manager - Manages system prompt, skills, and memory.

Usage:
    from agent.context import ContextManager

    context = ContextManager()
    system_prompt = context.build_system_prompt()
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from jinja2 import Template


@dataclass
class Skill:
    """Represents a skill loaded from markdown file."""
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    instructions: str = ""
    raw_content: str = ""


@dataclass
class Memory:
    """Represents a memory entry loaded from markdown file."""
    name: str
    content: str
    raw_content: str = ""


class ContextManager:
    """
    Manages system prompt assembly from skills, memory, and template.

    Directory structure:
        skills/              # Skill files (*.md)
        memory/              # Memory files (*.md)
        agent/prompts/       # Prompt templates
            system_prompt.md # System prompt template (Jinja2)

    Skill markdown format:
        ---
        name: skill_name
        description: Skill description
        triggers:
          - trigger condition 1
          - trigger condition 2
        ---
        # Instructions
        Actual skill instructions here...

    Memory markdown format:
        ---
        name: memory_name
        ---
        Memory content here...
    """

    def __init__(self, project_root: str = None):
        """
        Initialize ContextManager.

        Args:
            project_root: Path to project root. Defaults to parent of agent/
        """
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = Path(__file__).parent.parent

        self.skills_dir = self.project_root / "skills"
        self.memory_dir = self.project_root / "memory"
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt_template = self.prompts_dir / "system_prompt.md"

        self._skills: dict[str, Skill] = {}
        self._memory: dict[str, Memory] = {}
        self._template: Optional[str] = None
        self._loaded = False

    def load(self) -> None:
        """Load all skills, memory, and template from files."""
        self._load_skills()
        self._load_memory()
        self._load_template()
        self._loaded = True

    def _load_template(self) -> None:
        """Load system prompt template."""
        if self.system_prompt_template.exists():
            self._template = self.system_prompt_template.read_text(encoding="utf-8")
        else:
            self._template = None

    def _load_skills(self) -> None:
        """Load all skill files from skills directory."""
        self._skills.clear()

        if not self.skills_dir.exists():
            return

        for file_path in self.skills_dir.glob("*.md"):
            skill = self._parse_skill_file(file_path)
            if skill:
                self._skills[skill.name] = skill

    def _parse_skill_file(self, file_path: Path) -> Optional[Skill]:
        """Parse a skill markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._parse_skill_content(content)
        except Exception as e:
            print(f"Error loading skill {file_path}: {e}")
            return None

    def _parse_skill_content(self, content: str) -> Skill:
        """Parse skill content with frontmatter."""
        name = ""
        description = ""
        triggers = []
        instructions = ""

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                instructions = parts[2].strip()

                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line[5:].strip()
                    elif line.startswith("description:"):
                        description = line[12:].strip()
                    elif line.startswith("- "):
                        triggers.append(line[2:].strip())

        return Skill(
            name=name,
            description=description,
            triggers=triggers,
            instructions=instructions,
            raw_content=content,
        )

    def _load_memory(self) -> None:
        """Load all memory files from memory directory."""
        self._memory.clear()

        if not self.memory_dir.exists():
            return

        for file_path in self.memory_dir.glob("*.md"):
            memory = self._parse_memory_file(file_path)
            if memory:
                self._memory[memory.name] = memory

    def _parse_memory_file(self, file_path: Path) -> Optional[Memory]:
        """Parse a memory markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._parse_memory_content(content)
        except Exception as e:
            print(f"Error loading memory {file_path}: {e}")
            return None

    def _parse_memory_content(self, content: str) -> Memory:
        """Parse memory content with frontmatter."""
        name = ""
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                body = parts[2].strip()

                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line[5:].strip()

        return Memory(
            name=name,
            content=body,
            raw_content=content,
        )

    def get_skills(self) -> dict[str, Skill]:
        """Get all loaded skills."""
        if not self._loaded:
            self.load()
        return self._skills

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name."""
        if not self._loaded:
            self.load()
        return self._skills.get(name)

    def get_memory(self) -> dict[str, Memory]:
        """Get all loaded memory entries."""
        if not self._loaded:
            self.load()
        return self._memory

    def get_skills_info(self) -> str:
        """Get formatted skills information for system prompt."""
        if not self._loaded:
            self.load()

        if not self._skills:
            return ""

        lines = ["## 可用技能 (Skills)", ""]
        for name, skill in self._skills.items():
            lines.append(f"### {name}")
            if skill.description:
                lines.append(f"描述: {skill.description}")
            if skill.triggers:
                lines.append(f"触发条件: {', '.join(skill.triggers)}")
            lines.append("")

        return "\n".join(lines)

    def get_memory_info(self) -> str:
        """Get formatted memory information for system prompt."""
        if not self._loaded:
            self.load()

        if not self._memory:
            return ""

        lines = ["## 记忆 (Memory)", ""]
        for name, mem in self._memory.items():
            lines.append(f"### {name}")
            lines.append(mem.content)
            lines.append("")

        return "\n".join(lines)

    def get_skill_instructions(self, skill_name: str) -> str:
        """Get the instructions for a specific skill."""
        skill = self.get_skill(skill_name)
        if skill:
            return skill.instructions
        return ""

    def build_system_prompt(self) -> str:
        """
        Build the complete system prompt using Jinja2 template.

        The template is loaded from agent/prompts/system_prompt.md.
        Variables available in template:
            - skills: list of Skill objects
            - memory: list of Memory objects
        """
        if not self._loaded:
            self.load()

        if self._template:
            # Use Jinja2 template
            template = Template(self._template)
            return template.render(
                skills=list(self._skills.values()),
                memory=list(self._memory.values()),
            ).strip()
        else:
            # Fallback: build without template
            parts = ["你是一个智能助手，可以调用工具来获取信息或执行任务。"]

            if self._skills:
                parts.append(self.get_skills_info())
            if self._memory:
                parts.append(self.get_memory_info())

            return "\n\n".join(parts)

    def reload(self) -> None:
        """Reload all skills and memory from files."""
        self._loaded = False
        self.load()

    def __repr__(self) -> str:
        return f"ContextManager(skills={list(self._skills.keys())}, memory={list(self._memory.keys())})"
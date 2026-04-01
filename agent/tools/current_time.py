"""
当前时间工具 - 获取系统当前时间。

这是一个最简单的 BaseTool 示例，展示：
1. 继承 langchain_core.tools.BaseTool
2. 定义 name 和 description
3. 实现 _run 方法
"""
from datetime import datetime
from langchain_core.tools import BaseTool


class CurrentTimeTool(BaseTool):
    """获取当前系统时间的工具。"""

    # 工具名称，LLM 通过这个名字调用
    name: str = "current_time"

    # 工具描述，LLM 通过描述判断何时使用这个工具
    description: str = "获取当前系统时间。当用户询问时间、日期时使用。"

    def _run(self, timezone: str = "Asia/Shanghai") -> str:
        """
        执行工具逻辑。

        Args:
            timezone: 时区名称，默认北京时间

        Returns:
            格式化的当前时间字符串
        """
        # 获取当前时间并格式化
        now = datetime.now()
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
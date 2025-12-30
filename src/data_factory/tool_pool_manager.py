"""
Tool Pool Manager for AutoTool approach
Handles dynamic tool sampling to prevent overfitting
"""

import json
import random
from typing import List, Dict, Optional, Set
from pathlib import Path


class ToolPoolManager:
    """
    Manages dynamic tool sampling for AutoTool approach
    Ensures diverse tool exposure across training data
    """

    def __init__(self, tool_pool_path: str = "tools/tool_pool.json"):
        """Load tool pool from JSON"""
        with open(tool_pool_path, 'r', encoding='utf-8') as f:
            self.tool_pool = json.load(f)

        # Flatten tools by category
        self.all_tools = []
        self.categories = {}

        for category, tools in self.tool_pool.items():
            self.categories[category] = tools
            self.all_tools.extend(tools)

    def sample_tools_for_conversation(
        self,
        required_tools: Set[str],
        num_distractors: int = 4,
        category_aware: bool = True
    ) -> List[Dict]:
        """
        Sample tools for a single conversation

        Args:
            required_tools: Set of tool names that MUST be included (e.g., {"close_conversation"})
            num_distractors: Number of additional "distractor" tools to include
            category_aware: If True, prefer tools from same category as required tools

        Returns:
            List of tool schemas to expose in this conversation
        """
        selected_tools = []

        # 1. Add all required tools
        for tool_name in required_tools:
            tool = self._find_tool_by_name(tool_name)
            if tool:
                selected_tools.append(tool)

        # 2. Add distractor tools
        if category_aware:
            # Sample from same categories as required tools
            categories_to_sample = set()
            for tool_name in required_tools:
                category = self._find_category(tool_name)
                if category:
                    categories_to_sample.add(category)

            # Sample distractors from these categories
            available_tools = []
            for cat in categories_to_sample:
                available_tools.extend(self.categories[cat])

            # Remove already selected tools
            available_tools = [t for t in available_tools if t not in selected_tools]

            # Random sample
            num_to_sample = min(num_distractors, len(available_tools))
            if num_to_sample > 0:
                distractors = random.sample(available_tools, num_to_sample)
                selected_tools.extend(distractors)
        else:
            # Sample from entire pool
            available_tools = [t for t in self.all_tools if t not in selected_tools]
            num_to_sample = min(num_distractors, len(available_tools))
            if num_to_sample > 0:
                distractors = random.sample(available_tools, num_to_sample)
                selected_tools.extend(distractors)

        return selected_tools

    def _find_tool_by_name(self, tool_name: str) -> Optional[Dict]:
        """Find tool by name"""
        for tool in self.all_tools:
            if tool['name'] == tool_name:
                return tool
        return None

    def _find_category(self, tool_name: str) -> Optional[str]:
        """Find category of a tool"""
        for category, tools in self.categories.items():
            for tool in tools:
                if tool['name'] == tool_name:
                    return category
        return None

    def get_tool_selection_guidance(self, tools: List[Dict]) -> str:
        """
        Generate guidance text for tool selection reasoning

        Args:
            tools: List of tools available in this conversation

        Returns:
            String describing available tools and when to use them
        """
        guidance = "## AVAILABLE TOOLS IN THIS CONVERSATION:\n\n"

        for i, tool in enumerate(tools, 1):
            guidance += f"{i}. **{tool['name']}**\n"
            guidance += f"   - Description: {tool['description']}\n"

            if 'use_cases' in tool:
                guidance += f"   - Use when: {', '.join(tool['use_cases'])}\n"

            guidance += "\n"


        return guidance


# Example usage
if __name__ == "__main__":
    manager = ToolPoolManager()

    # Sample tools for a conversation that needs to close
    tools = manager.sample_tools_for_conversation(
        required_tools={"close_conversation"},
        num_distractors=2,
        category_aware=True
    )

    print("Sampled tools:")
    for tool in tools:
        print(f"  - {tool['name']}")

    print("\n" + manager.get_tool_selection_guidance(tools))

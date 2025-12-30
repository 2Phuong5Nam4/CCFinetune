"""
Reconstruction Pipeline: Raw Conversations → Annotated Training Data
Using LangChain + OpenAI to add <think> tags and <tool_call> structures
"""

import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
from dataclasses import dataclass

from src.data_factory.tool_pool_manager import ToolPoolManager

# Load environment variables
load_dotenv()







# ============================================================================
# Output Schema for Structured Parsing
# ============================================================================

class SingleTurn(BaseModel):
    """
    Schema for one assistant turn with reasoning and optional tool call.

    PATTERN 1 (React Agent):
    - If tool_call is present: Turn chỉ có thinking + tool_call, KHÔNG có response
    - If tool_call is null: Turn chỉ có thinking + response
    """

    thinking: str = Field(
        description="""Quá trình suy nghĩ nội tâm, BAO GỒM:
        1. Phân tích tình huống hiện tại
        2. Xác định nếu cần tool (yes/no and why)
            - Nếu có, liệt kê tất cả tools có sẵn
            - Giải thích tại sao chọn tool X
        """
    )

    tool_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cấu trúc gọi công cụ (nếu có tool_call thì response = null)"
    )

   

# ============================================================================
# Reconstruction Pipeline
# ============================================================================

class ConversationReconstructor:
    """
    Uses Teacher LLM to reconstruct reasoning and tool calls
    from raw conversation history
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.3,
        tool_pool_path: str = "tools/tool_pool.json"
    ):
        print("model_name:", model_name)
        self.llm = create_agent(
            model=ChatOpenAI(
                model=model_name,
                temperature=temperature,
            ),
            response_format=SingleTurn  # Changed from ReconstructedConversation
        )
        self.tool_pool_manager = ToolPoolManager(tool_pool_path=tool_pool_path)


    def create_reconstruction_prompt(self, sampled_tools: List[Dict]) -> str:
        """
        Create prompt template for Teacher LLM with dynamically sampled tools

        Args:
            sampled_tools: Tools sampled for this specific conversation
        """

        # Generate tool-specific guidance
        tool_guidance = self.tool_pool_manager.get_tool_selection_guidance(sampled_tools)

        system_prompt = f"""Bạn là chuyên gia phân tích cuộc hội thoại khảo sát NPS của HEINEKEN Việt Nam.

{tool_guidance}

# NHIỆM VỤ
Phân tích conversation history và tạo **CHỈ MỘT TURN** cho assistant tiếp theo:
- **thinking**: Phân tích ngắn gọn tình huống + hành động (1-3 câu)
- **tool_call**: Gọi tool nếu cần (optional)

**LƯU Ý**: Response text đã có sẵn, bạn CHỈ tạo thinking + tool_call.

---

# QUY TẮC THINKING

## Cấu trúc bắt buộc (1-3 câu):

**Khi KHÔNG cần tool:**
```
"[Tình huống]. [Bước tiếp theo]."
```

**Khi CẦN tool:**
```
"[Tình huống]. [Cần tool]. Tools: [A, B, C]. → Chọn [X] vì [lý do]."
```

## Ví dụ cụ thể:

✅ **GOOD** (ngắn gọn, đúng trọng tâm):
```
"KH xác nhận thông tin đúng. Giới thiệu mục đích khảo sát."

"KH cho 7 điểm (bình thường). Hỏi cần cải thiện gì để lên 10."

"KH hoàn thành khảo sát, không có ý kiến thêm. Cần kết thúc. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn close_conversation vì survey xong."

"Tool close_conversation đã execute. Gửi lời cảm ơn."
```

❌ **BAD** (dài dòng, không cần thiết):
```
"Khách hàng đã xác nhận thông tin là chính xác. Điều này cho thấy chúng ta đang nói chuyện với đúng người. Bây giờ tôi cần chuyển sang bước tiếp theo trong quy trình là giới thiệu mục đích của cuộc gọi này..."
```

---

# QUY TRÌNH KHẢO SÁT - 4 BƯỚC

## 1. Mở đầu
- Chào + giới thiệu + xác minh SĐT/tên KH

**Thinking mẫu:**
```
"Bắt đầu cuộc gọi với KH [tên]. Chào hỏi và xác minh thông tin."
"KH xác nhận đúng. Giới thiệu mục đích khảo sát."
```

---

## 2. Mục đích
- Giải thích khảo sát độ hài lòng (KHÔNG phải sales)
- Xin phép trao đổi

**Thinking mẫu:**
```
"KH đồng ý tham gia. Hỏi điểm NPS 0-10."
```

---

## 3. Khảo sát NPS

### Câu 1: Điểm 0-10
**Thinking mẫu:**
```
"KH cho [X] điểm ([nhóm]). [Hành động theo nhóm]."
```
- 0-6 → "Xin lỗi và hỏi điều gì chưa hài lòng"
- 7-8 → "Hỏi cần cải thiện gì để lên 10"
- 9-10 → "Hỏi hài lòng điều gì nhất"

### Câu 2-3: Khai thác chi tiết
**Thinking mẫu:**
```
"KH phàn nàn [vấn đề]. Ghi nhận và hỏi tiêu chí khác."
"KH khen [điểm mạnh]. Ghi nhận và hỏi còn góp ý gì không."
```

---

## 4. Kết thúc - TOOL CALL

### Pattern 2 turns:

**Turn N - Gọi tool:**
```json
{{
  "thinking": "KH hoàn thành khảo sát. Cần kết thúc. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn close_conversation vì survey xong.",
  "tool_call": {{
    "name": "close_conversation",
    "parameters": {{"close_message": "Survey completed"}}
  }}
}}
```

**Turn N+1 - Sau tool execute:**
```json
{{
  "thinking": "Tool đã execute. Gửi lời cảm ơn.",
  "tool_call": null
}}
```

---

# VÍ DỤ ĐẦY ĐỦ VỀ TOOL CALL

## Ví dụ 1: Kết thúc khảo sát thành công

**Conversation history:**
```
User: "Không em, thế thôi. Cảm ơn em nhé."
```

**Turn N - Gọi tool:**
```json
{{
  "thinking": "KH đã hoàn thành khảo sát và không có ý kiến thêm. Cần kết thúc cuộc gọi. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn close_conversation vì survey hoàn thành, KH hài lòng.",
  "tool_call": {{
    "name": "close_conversation",
    "parameters": {{
      "close_message": "Survey completed successfully"
    }}
  }}
}}
```

**Tool execution result:**
```json
{{
  "status": "success",
  "message": "Conversation closed"
}}
```

**Turn N+1 - Response sau tool:**
```json
{{
  "thinking": "Tool close_conversation đã thực thi thành công. Gửi lời cảm ơn cuối cùng cho KH.",
  "tool_call": null
}}
```

---

## Ví dụ 2: KH bận, cần hẹn lại

**Conversation history:**
```
User: "Em ơi anh đang bận, để anh gọi lại sau được không?"
```

**Turn N - Gọi tool:**
```json
{{
  "thinking": "KH đang bận, chưa hoàn thành khảo sát. Cần tạm dừng và hẹn gọi lại. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn pause_conversation vì cần tiếp tục sau.",
  "tool_call": {{
    "name": "pause_conversation",
    "parameters": {{
      "reason": "Customer busy, will call back later",
      "callback_time": "later"
    }}
  }}
}}
```

**Turn N+1 - Response sau tool:**
```json
{{
  "thinking": "Tool pause_conversation đã execute. Xin lỗi và xác nhận sẽ gọi lại sau.",
  "tool_call": null
}}
```

---

## Ví dụ 3: KH yêu cầu chuyển nhân viên khác

**Conversation history:**
```
User: "Em cho anh nói chuyện với quản lý được không? Anh có vấn đề muốn phản ánh."
```

**Turn N - Gọi tool:**
```json
{{
  "thinking": "KH có vấn đề cần phản ánh, yêu cầu nói với quản lý. Không thể xử lý qua khảo sát. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn transfer_to_human vì cần chuyên viên cấp cao.",
  "tool_call": {{
    "name": "transfer_to_human",
    "parameters": {{
      "reason": "Customer requests to speak with manager",
      "issue": "Escalation needed"
    }}
  }}
}}
```

**Turn N+1 - Response sau tool:**
```json
{{
  "thinking": "Tool transfer_to_human đã execute. Xác nhận sẽ chuyển sang bộ phận phù hợp.",
  "tool_call": null
}}
```

---

## Ví dụ 4: Không cần tool - Turn bình thường

**Conversation history:**
```
User: "Dạ 8 điểm ạ"
```

**Output:**
```json
{{
  "thinking": "KH cho 8 điểm (bình thường). Hỏi cần cải thiện gì để lên 10 điểm.",
  "tool_call": null
}}
```

---

## Ví dụ 5: KH từ chối tham gia

**Conversation history:**
```
User: "Anh không muốn khảo sát, anh bận lắm."
```

**Turn N - Gọi tool:**
```json
{{
  "thinking": "KH từ chối tham gia khảo sát. Cần kết thúc cuộc gọi. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn close_conversation vì KH không muốn tiếp tục.",
  "tool_call": {{
    "name": "close_conversation",
    "parameters": {{
      "close_message": "Customer declined to participate"
    }}
  }}
}}
```

**Turn N+1 - Response sau tool:**
```json
{{
  "thinking": "Tool đã execute. Xin lỗi và chúc KH một ngày tốt lành.",
  "tool_call": null
}}
```

---

## Ví dụ 6: KH số điện thoại sai

**Conversation history:**
```
User: "Không phải, anh không phải người em tìm đâu."
```

**Turn N - Gọi tool:**
```json
{{
  "thinking": "Gọi nhầm số. Cần xin lỗi và kết thúc cuộc gọi. Tools: close_conversation, pause_conversation, transfer_to_human. → Chọn close_conversation vì sai thông tin KH.",
  "tool_call": {{
    "name": "close_conversation",
    "parameters": {{
      "close_message": "Wrong contact information"
    }}
  }}
}}
```

**Turn N+1 - Response sau tool:**
```json
{{
  "thinking": "Tool đã execute. Xin lỗi KH vì nhầm lẫn.",
  "tool_call": null
}}
```

---

# KHI NÀO GỌI TOOL?

**✅ GỌI `close_conversation` khi:**
- KH nói "Không có gì thêm" / "Thế thôi" / "OK em"
- KH bận, KHÔNG muốn tiếp tục (khác với "bận tạm thời")
- Đã hỏi xong tất cả câu khảo sát
- Gọi nhầm số / sai thông tin KH
- KH từ chối tham gia


**✅ GỌI `transfer_to_human` khi:**
- KH có vấn đề nghiêm trọng cần leo thang
- KH yêu cầu nói chuyện với quản lý/chuyên viên
- Vấn đề vượt quá khả năng xử lý của khảo sát

**❌ KHÔNG gọi tool khi:**
- Còn đang hỏi điểm / khai thác ý kiến
- KH đang chia sẻ thông tin
- Chưa xong quy trình
- Đang giải thích/làm rõ thông tin

---

# CRITICAL: NGUYÊN TẮC THINKING NGẮN GỌN

1. **Tối đa 3 câu**, thường chỉ cần 1-2 câu
2. **Tập trung vào**: Tình huống hiện tại + Hành động tiếp theo
3. **Khi gọi tool**: PHẢI giải thích tại sao chọn tool đó
4. **Tránh**:
   - Lặp lại thông tin KH đã nói
   - Giải thích dài dòng về quy trình
   - Suy nghĩ nhiều bước phía trước
   - Phân tích tâm lý KH quá sâu

---

# OUTPUT FORMAT
```json
{{
  "thinking": "1-3 câu ngắn gọn, đúng trọng tâm",
  "tool_call": null | {{
    "name": "tool_name",
    "parameters": {{...}}
  }}
}}
```

---

Phân tích conversation history và tạo thinking + tool_call cho TURN TIẾP THEO:"""

        return system_prompt

    


    async def reconstruct_conversation(
        self,
        raw_conversation: Dict,
    ) -> Dict:
        """
        Main reconstruction function - generates thinking and tool_calls for each turn

        Args:
            raw_conversation: Dict with 'messages' key containing conversation

        Returns:
            Annotated conversation with <think> tags and <tool_call> structures

        Raises:
            ValueError: If validation fails
        """
        messages = raw_conversation['messages']

        # 1. Sample tools for this conversation
        sampled_tools = self.tool_pool_manager.sample_tools_for_conversation(
            required_tools={"close_conversation"},
            num_distractors=4,
            category_aware=True
        )

        # 2. Create system prompt with sampled tools
        system_prompt = self.create_reconstruction_prompt(sampled_tools)

        # 3. Construct annotated conversation step by step
        constructed_conversation = []

        for message in messages:
            if message['role'] == 'system':
                constructed_conversation.append(message)
                continue

            elif message['role'] == 'user':
                constructed_conversation.append(message)
                continue

            elif message['role'] == 'assistant':
                # Generate thinking + tool_call (if needed) for this turn
                response = await self.llm.ainvoke({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(constructed_conversation, ensure_ascii=False)},
                    ]
                })
                response = response["structured_response"]

                # Check if tool_call is present in response
                if response.tool_call is not None:
                    print("Detected tool_call:", response.tool_call)
                    # Turn 1: thinking + tool_call (NO response text)
                    constructed_conversation.append({
                        "role": "assistant",
                        "content": f"<think>{response.thinking}</think>\n<tool_call>{json.dumps(response.tool_call, ensure_ascii=False)}</tool_call>"
                    })

                    # Simulate tool execution result
                    constructed_conversation.append({
                        "role": "tool",
                        "name": response.tool_call["name"],
                        "content": json.dumps({
                            "status": "success",
                            "message": f"Tool {response.tool_call['name']} executed successfully."
                        }, ensure_ascii=False)
                    })

                    # Turn 2: Generate thinking for response after tool execution
                    post_tool_response = await self.llm.ainvoke({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(constructed_conversation, ensure_ascii=False)},
                        ]
                    })
                    post_tool_response = post_tool_response["structured_response"]

                    # Add final response turn with thinking + response text
                    constructed_conversation.append({
                        "role": "assistant",
                        "content": f"<think>{post_tool_response.thinking}</think>\n{message['content']}"
                    })

                else:
                    # Normal turn: thinking + response text (NO tool_call)
                    constructed_conversation.append({
                        "role": "assistant",
                        "content": f"<think>{response.thinking}</think>\n{message['content']}"
                    })

        return {"messages": constructed_conversation}
# ============================================================================
# Batch Processing
# ============================================================================

async def reconstruct_dataset(
    input_path: str,
    output_path: str,
    model_name: str = "gpt-4o",
    max_samples: Optional[int] = None,
    max_concurrent: int = 3
):
    """
    Process entire dataset with parallel processing

    Args:
        input_path: Path to raw conversations.jsonl
        output_path: Path to save annotated conversations
        model_name: Teacher LLM model name
        max_samples: Limit number of samples (for testing)
        max_concurrent: Maximum number of conversations to process in parallel (default: 3)
    """
    reconstructor = ConversationReconstructor(model_name=model_name)

    # Load raw conversations
    conversations = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    if max_samples:
        conversations = conversations[:max_samples]

    print(f"Processing {len(conversations)} conversations with max {max_concurrent} parallel tasks...")

    # Semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Results storage
    results = []
    failed_count = 0

    async def process_single_conversation(idx: int, conv: Dict) -> tuple[int, Optional[Dict], Optional[str]]:
        """Process a single conversation with semaphore control"""
        async with semaphore:
            try:
                print(f"[{idx+1}/{len(conversations)}] Processing conversation...")
                annotated = await reconstructor.reconstruct_conversation(conv)
                return (idx, annotated, None)
            except Exception as e:
                error_msg = f"Error processing conversation {idx}: {str(e)}"
                print(f"❌ {error_msg}")
                return (idx, None, error_msg)

    # Create tasks for all conversations
    tasks = [
        process_single_conversation(idx, conv)
        for idx, conv in enumerate(conversations)
    ]

    # Execute all tasks concurrently (limited by semaphore)
    task_results = await asyncio.gather(*tasks)

    # Collect successful results in order
    annotated_conversations = []
    for idx, annotated, error in sorted(task_results, key=lambda x: x[0]):
        if annotated is not None:
            annotated_conversations.append(annotated)
        else:
            failed_count += 1

    # Save results
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in annotated_conversations:
            # Convert ReconstructedConversation to message format
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"\n✅ Completed!")
    print(f"   Success: {len(annotated_conversations)}")
    print(f"   Failed: {failed_count}")
    print(f"   Saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(reconstruct_dataset(
        input_path="data/raw/conversations.jsonl",
        output_path="data/processed/annotated_conversations.jsonl",
        model_name="gpt-4o",
        max_samples=2,  # Test on 2 samples first
        max_concurrent=3  # Process 3 conversations in parallel
    ))

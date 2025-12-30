 ---
  PHASE 1: Setup Môi Trường & Cấu Trúc Project

  Bước 1.1: Tạo cấu trúc thư mục

  CCFinetune/
  ├── configs/          # File config YAML
  ├── data/
  │   ├── raw/         # Dữ liệu gốc (conversation history, SOP)
  │   ├── processed/   # Dữ liệu đã reconstruct
  │   └── datasets/    # Dataset final cho training
  ├── tools/           # Tool schemas JSON
  ├── src/
  │   ├── data_factory/
  │   ├── training/
  │   ├── rewards/
  │   └── inference/
  ├── scripts/         # Script chạy training
  ├── outputs/         # Checkpoints, logs
  └── tests/          # Unit tests

  Bước 1.2: Setup môi trường Python

  - Tạo virtual environment (Python 3.10+)
  - Cài đặt dependencies:
    - unsloth (fine-tuning optimization)
    - transformers, trl, peft
    - torch (CUDA compatible)
    - datasets, accelerate
    - OpenAI/Anthropic SDK (cho Teacher LLM)

  Bước 1.3: Tải base model

  - Download unsloth/Llama-3.1-8B-Instruct hoặc model khác
  - Test load model với 4-bit quantization

  ---
  PHASE 2: Định Nghĩa Tools từ SOP

  Bước 2.1: Phân tích SOP documents

  - Đọc toàn bộ SOP của quy trình customer service
  - Xác định các hành động mà agent thực hiện (tra cứu đơn hàng, tạo yêu cầu hoàn tiền, kiểm tra tồn kho...)

  Bước 2.2: Thiết kế tool schemas

  - Tạo file tools/tool_schemas.json
  - Mỗi tool cần:
    - name: Tên function
    - description: Mô tả rõ ràng khi nào dùng
    - parameters: JSON schema (type, properties, required fields)

  Bước 2.3: Validate tools

  - Kiểm tra schema có cover đủ các use cases trong SOP không
  - Có thể dùng Teacher LLM để auto-extract từ SOP rồi manual review

  ---
  PHASE 3: Reconstruction Pipeline - Tạo Dữ Liệu Training

  Bước 3.1: Chuẩn bị raw data

  - Thu thập lịch sử conversation từ hệ thống hiện tại
  - Format về dạng chuẩn (mảng messages với role/content)
  - Lưu vào data/raw/conversations.jsonl

  Bước 3.2: Implement reconstruction script

  Tạo src/data_factory/reconstruction.py:

  Chức năng chính:
  - Input: Raw conversation + SOP + Tool schemas
  - Gọi Teacher LLM (GPT-4o/Claude) để infer:
    - Reasoning (<think> tags): Agent đang suy luận gì?
    - Tool calls (<tool_call> JSON): Gọi tool nào với params gì?
    - Tool results (<tool_result>): Kết quả trả về là gì (infer từ câu trả lời tiếp theo)
  - Output: Annotated conversation với đầy đủ reasoning + tool calls

  Prompt template cho Teacher LLM:
  - Cung cấp tools schema
  - Cung cấp SOP
  - Cung cấp raw conversation
  - Yêu cầu reconstruct từng assistant turn

  Bước 3.3: Implement validation pipeline

  Tạo src/data_factory/validation.py:

  Kiểm tra:
  - Format đúng không? (<think>, <tool_call> paired tags)
  - JSON trong <tool_call> valid không?
  - Tool call/result matching (mỗi call có result tương ứng?)
  - Reasoning có consistent với action không?
  - Loại bỏ samples lỗi

  Bước 3.4: Chạy reconstruction

  - Chạy trên toàn bộ raw conversations
  - Monitor success rate
  - Manual spot-check một vài samples để verify quality
  - Target: Ít nhất 500-1000 conversations chất lượng cao

  Bước 3.5: Format cho training

  Chuyển đổi sang format Llama-3.1 chat template:
  <|start_header_id|>system<|end_header_id|>
  {system_prompt}<|eot_id|>

  <|start_header_id|>user<|end_header_id|>
  {user_message}<|eot_id|>

  <|start_header_id|>assistant<|end_header_id|>
  <think>{reasoning}</think>
  <tool_call>{json}</tool_call><|eot_id|>

  <|start_header_id|>tool<|end_header_id|>
  {tool_result}<|eot_id|>

  ...

  Lưu vào data/datasets/sft_train.jsonl

  ---
  PHASE 4: SFT Training

  Bước 4.1: Tạo config file

  configs/sft_config.yaml:
  - Model name
  - LoRA parameters (r=64, alpha=16, target_modules)
  - Training args (batch size, learning rate, steps)
  - Max sequence length (8192)

  Bước 4.2: Setup data collator

  Sử dụng DataCollatorForCompletionOnlyLM:
  - Response template: "<|start_header_id|>assistant<|end_header_id|>"
  - Auto-mask toàn bộ system/user/tool messages
  - Chỉ train trên assistant responses (bao gồm <think> và <tool_call>)

  Bước 4.3: Tạo training script

  scripts/train_sft.py:
  - Load model với Unsloth (4-bit quantization)
  - Apply LoRA
  - Load dataset
  - Setup SFTTrainer
  - Training loop
  - Save checkpoint

  Bước 4.4: Verify masking

  Trước khi train, chạy script verify masking:
  - Kiểm tra labels có mask đúng không (-100 cho non-assistant)
  - Đảm bảo ~30% tokens được train, 70% masked
  - Xác nhận tất cả assistant turns đều được unmask

  Bước 4.5: Training

  - Chạy training với eval mỗi 100 steps
  - Monitor loss curve (phải giảm dần)
  - Save best checkpoint
  - Thời gian dự kiến: Vài giờ trên GPU (A100/H100) hoặc 1-2 ngày trên consumer GPU

  Bước 4.6: Evaluation

  Test model trên một vài prompts thủ công:
  - Model có generate <think> tags không?
  - Model có call tools đúng format không?
  - Reasoning có hợp lý không?

  ---
  PHASE 5: GRPO Training - Refine Reasoning

  Bước 5.1: Chuẩn bị prompts dataset

  - Tạo data/datasets/grpo_prompts.jsonl
  - Mỗi item là initial user message (không bao gồm assistant response)
  - Đảm bảo đa dạng các scenarios

  Bước 5.2: Implement reward functions

  src/rewards/:

  format_reward.py:
  - Kiểm tra tags đúng format
  - JSON valid
  - Tool calls theo schema

  correctness_reward.py:
  - So sánh với ground truth (nếu có)
  - Hoặc dùng LLM-as-Judge
  - Rule-based validation (tool sequence logic)

  efficiency_reward.py:
  - Đếm số turns
  - Đếm số tool calls redundant
  - Token efficiency

  style_reward.py:
  - Embedding similarity với reference examples
  - Regex penalties (tone, formality)
  - LLM judge cho naturalness

  Bước 5.3: Implement generation với diversity

  Setup sampling parameters:
  - temperature=0.9 (hoặc 0.7-1.2)
  - top_p=0.9
  - top_k=50
  - Seed randomization cho mỗi generation

  Generate 8 responses per prompt (G=8):
  - Mỗi response là một full conversation (có thể 3-10 turns)
  - Phải execute tools thật hoặc mock tool results

  Bước 5.4: Implement diversity metrics

  Monitor:
  - Tool diversity ratio: unique_tool_sequences / total_sequences
  - Text similarity: Average cosine similarity giữa các generations
  - Target: Tool diversity > 0.6, text similarity < 0.7

  Bước 5.5: Tạo GRPO training script

  scripts/train_grpo.py:
  - Load SFT checkpoint
  - For each prompt:
    - Generate G=8 responses
    - Compute rewards cho mỗi response
    - Rank responses
    - Compute GRPO loss
    - Update policy
  - Monitor diversity metrics mỗi iteration

  Bước 5.6: Training

  - Chạy 200-500 iterations
  - Monitor:
    - Average rewards tăng
    - Diversity metrics stable
    - Generated samples quality
  - Early stopping nếu diversity collapse

  ---
  PHASE 6: Merge & Deployment

  Bước 6.1: Merge LoRA adapters

  - Script scripts/merge_lora.py
  - Merge LoRA weights vào base model
  - Save merged model

  Bước 6.2: Quantization (optional)

  - Quantize về GGUF hoặc AWQ format
  - Để deploy trên resource-constrained environment

  Bước 6.3: Tạo inference pipeline

  src/inference/agent_runner.py:
  - Load merged model
  - Implement tool execution logic
  - Streaming generation
  - Parse <think>, <tool_call>, final response

  Bước 6.4: Testing

  - Test trên real user queries
  - So sánh với baseline (GPT-4, human agents)
  - Metrics: Correctness, user satisfaction, efficiency

  ---
  PHASE 7: Iteration & Monitoring

  Bước 7.1: Collect production data

  - Log conversations từ deployed model
  - Identify failure cases

  Bước 7.2: Data flywheel

  - Annotate new conversations
  - Add vào training set
  - Re-train (continual learning)

  Bước 7.3: A/B testing

  - So sánh versions khác nhau
  - Optimize hyperparameters (LoRA rank, GRPO weights)

  ---
  Timeline Tổng Quan:

  1. Phase 1-2: 2-3 ngày (setup + tool definition)
  2. Phase 3: 1-2 tuần (reconstruction + validation)
  3. Phase 4: 3-5 ngày (SFT training + eval)
  4. Phase 5: 1 tuần (GRPO implementation + training)
  5. Phase 6: 2-3 ngày (deployment)
  6. Phase 7: Ongoing

  ---
  Critical Success Factors:

  1. Data quality > Data quantity: 500 high-quality conversations tốt hơn 5000 noisy conversations
  2. Validation ở mọi bước: Đừng train trên garbage data
  3. Start small: Test trên subset nhỏ trước, scale sau
  4. Monitor diversity: GRPO chỉ hiệu quả nếu có diversity
  5. Iteration nhanh: Đừng chờ perfect, ship rồi improve
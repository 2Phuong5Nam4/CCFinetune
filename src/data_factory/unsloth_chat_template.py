from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import os

def prepare_qwen_dataset(file_path, model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", max_seq_length=2048):
    """
    Load dataset từ JSONL và apply Qwen chat template.
    """
    # 1. Khởi tạo Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    # 2. Cấu hình Template Qwen-2.5
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    # 3. Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")

    # 4. Load và Format dữ liệu
    dataset = load_dataset("json", data_files=file_path, split="train")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        # tokenize=False để lấy chuỗi text đã format (dùng cho SFTTrainer)
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset, tokenizer

# --- Thực thi ---
DATA_PATH = "/home/PhuongNam/CompanyProject/CCFinetune/data/processed/annotated_conversations.jsonl"

try:
    formatted_dataset, tokenizer = prepare_qwen_dataset(DATA_PATH)

    # In vài sample để kiểm tra
    print("\n" + "="*30 + " KIỂM TRA DỮ LIỆU " + "="*30)
    for i in range(2):
        print(f"\n[SAMPLE {i+1}]")
        print(formatted_dataset[i]["text"])
        print("-" * 80)

except Exception as e:
    print(f"Lỗi: {e}")
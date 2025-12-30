# **Báo Cáo Kỹ Thuật Chuyên Sâu: Kiến Trúc Agentic Fine-tuning và Triển Khai Chatbot L1 Sử Dụng Unsloth và GRPO**

## **Tóm tắt Điều hành**

Sự chuyển dịch của trí tuệ nhân tạo doanh nghiệp vào cuối năm 2025 đang đánh dấu một bước ngoặt quan trọng từ các hệ thống chatbot thụ động sang các mô hình tác nhân (agentic models) có khả năng tự chủ cao. Yêu cầu đặt ra không còn dừng lại ở việc truy xuất thông tin (RAG), mà là xây dựng các Chatbot Cấp độ 1 (L1) có khả năng tư duy như một nhân viên chính thức: hiểu sâu sắc thuật ngữ chuyên ngành, lập kế hoạch xử lý quy trình (SOP), ánh xạ kế hoạch đó thành các lệnh gọi công cụ (tool calls) chính xác, và giao tiếp với văn phong mang đậm bản sắc doanh nghiệp.1

Báo cáo này cung cấp một phân tích kỹ thuật toàn diện về quy trình "Agentic Fine-tuning" (Tinh chỉnh tác nhân), tập trung vào việc sử dụng thư viện **Unsloth** để tối ưu hóa hiệu suất phần cứng. Unsloth đóng vai trò then chốt nhờ khả năng giảm mức tiêu thụ VRAM tới 80% và tăng tốc độ huấn luyện lên gấp 2 lần, cho phép triển khai các kỹ thuật tiên tiến như **Group Relative Policy Optimization (GRPO)** ngay cả trên hạ tầng GPU hạn chế.3 Báo cáo sẽ đi sâu vào ba trụ cột chính: (1) Xây dựng "Nhà máy dữ liệu" (Data Factory) để chuyển đổi tài liệu tĩnh thành chuỗi tư duy động, (2) Thiết lập pipeline huấn luyện kết hợp SFT và GRPO để định hình tư duy và văn phong, và (3) Chiến lược triển khai thực tế trên Unsloth.

## ---

**1\. Cơ Sở Lý Luận và Kiến Trúc Của Chatbot L1**

### **1.1 Sự Chuyển Dịch Từ Truy Xuất (Retrieval) Sang Thực Thi (Execution)**

Trong các hệ thống AI doanh nghiệp truyền thống, mô hình RAG (Retrieval-Augmented Generation) đóng vai trò chủ đạo trong việc cung cấp kiến thức. Tuy nhiên, RAG bộc lộ những hạn chế nghiêm trọng khi đối mặt với yêu cầu về *quy trình* và *hành động*. Một mô hình RAG có thể tìm thấy tài liệu quy định về "Quy trình hoàn tiền", nhưng thường thất bại trong việc tự động phân rã quy trình đó thành các bước thực thi tuần tự hoặc thất bại trong việc duy trì giọng điệu chuyên nghiệp của một nhân viên chăm sóc khách hàng xuyên suốt cuộc hội thoại.4

Chatbot L1 được định nghĩa bởi khả năng "Suy nghĩ, Lập kế hoạch và Hành động" (Think, Plan, Act). Quá trình tinh chỉnh (fine-tuning) không chỉ đơn thuần là nạp kiến thức, mà là thay đổi trọng số của mô hình để hình thành một *trực giác quy trình*. Điều này cho phép mô hình:

1. **Hiểu thuật ngữ nội bộ:** Nắm bắt các từ viết tắt, mã dự án và biệt ngữ chuyên ngành mà không cần giải thích ngữ cảnh liên tục.6  
2. **Lập kế hoạch suy luận (Chain-of-Thought \- CoT):** Tự động sinh ra các chuỗi suy nghĩ nội bộ để phân tích vấn đề trước khi đưa ra câu trả lời cuối cùng, tương tự như cách con người tư duy.7  
3. **Đồng bộ hóa văn phong (Persona Alignment):** Phản hồi với giọng điệu, cấu trúc câu và thái độ phù hợp với văn hóa công ty, điều mà prompt engineering khó duy trì ổn định.9

### **1.2 Vai Trò Của Unsloth Trong Kỷ Nguyên Agentic AI**

Việc huấn luyện các mô hình có khả năng suy luận và gọi công cụ đòi hỏi tài nguyên tính toán khổng lồ, đặc biệt là khi áp dụng Reinforcement Learning (RL). Unsloth nổi lên như một giải pháp hạ tầng thiết yếu nhờ vào việc viết lại các hạt nhân (kernels) tính toán của PyTorch bằng ngôn ngữ Triton và thực hiện đạo hàm thủ công (manual autograd).10

Phân tích kỹ thuật cho thấy Unsloth mang lại hai lợi thế chiến lược cho việc xây dựng Chatbot L1:

* **Hiệu quả bộ nhớ cho ngữ cảnh dài:** Để mô hình có thể "suy nghĩ" (tạo ra các token \<think\>), ngữ cảnh huấn luyện thường phải rất dài (từ 8k đến 128k token). Unsloth giảm mức sử dụng VRAM xuống 60-80% thông qua việc tối ưu hóa Flash Attention và quản lý bộ nhớ đệm KV (Key-Value), cho phép huấn luyện các mô hình Llama-3 hoặc Mistral với ngữ cảnh dài trên một GPU đơn lẻ.3  
* **Hỗ trợ GRPO native:** Unsloth cung cấp các bản cài đặt tối ưu cho thuật toán GRPO, loại bỏ nhu cầu về một mô hình "Critic" (Phê bình) riêng biệt vốn tiêu tốn nhiều bộ nhớ trong các phương pháp PPO truyền thống, từ đó dân chủ hóa việc huấn luyện mô hình suy luận.12

## ---

**2\. Chiến Lược Dữ Liệu: Xây Dựng "Data Factory" Cho Agent**

Chất lượng của một agent được quyết định bởi chất lượng của dữ liệu huấn luyện. Đối với mục tiêu xây dựng một nhân viên ảo, dữ liệu thô từ các file PDF hay tài liệu hướng dẫn vận hành (SOP) là không đủ. Chúng ta cần thiết lập một quy trình chuyển đổi dữ liệu, hay còn gọi là "Data Factory", để biến các văn bản tĩnh thành các mẫu huấn luyện chứa đựng *logic suy luận* và *cấu trúc hành động*.

### **2.1 Cấu Trúc Dữ Liệu Mục Tiêu (Target Data Topology)**

Dữ liệu đầu vào cho quá trình fine-tuning trên Unsloth cần tuân thủ định dạng JSONL, nhưng nội dung bên trong phải chứa đựng ba thành phần cốt lõi để đạt được mục tiêu đề ra: Tư duy (Reasoning), Gọi công cụ (Tool Call), và Văn phong (Style).

Bảng dưới đây mô tả cấu trúc của một mẫu dữ liệu lý tưởng cho Chatbot L1:

| Thành phần | Định dạng kỹ thuật | Mục đích huấn luyện |
| :---- | :---- | :---- |
| **System Prompt** | Chứa định nghĩa Tools (JSON Schema) và chỉ thị về Persona. | Thiết lập không gian hành động và vai trò của agent.14 |
| **User Query** | Câu hỏi hoặc yêu cầu mô phỏng từ người dùng thực tế. | Tạo ngữ cảnh đầu vào đa dạng (đủ thông tin, thiếu thông tin, mơ hồ).16 |
| **Reasoning Trace** | Đặt trong thẻ \<think\>...\</think\>. | Dạy mô hình cách phân tích SOP, kiểm tra điều kiện và lập kế hoạch trước khi trả lời.8 |
| **Tool Call** | Định dạng JSON hoặc Python function call. | Dạy mô hình ánh xạ kế hoạch thành hành động máy tính chính xác.17 |
| **Final Response** | Văn bản tự nhiên bao quanh tool call. | Huấn luyện văn phong, giọng điệu và cách giao tiếp của nhân viên.9 |

### **2.2 Pipeline Tổng Thể: Từ SOP Đến Dataset**

Quy trình xây dựng dữ liệu không nên thực hiện thủ công mà cần được tự động hóa thông qua một pipeline sử dụng các mô hình ngôn ngữ lớn hơn (Teacher Models) để sinh dữ liệu cho mô hình nhỏ hơn (Student Models).

#### **Giai đoạn 1: Phân rã và Chunking Tài liệu (Ingestion)**

Bước đầu tiên là xử lý các tài liệu quy trình (SOP, Policy PDF). Việc cắt nhỏ văn bản (chunking) không thể thực hiện ngẫu nhiên theo số lượng token mà phải dựa trên *ngữ nghĩa quy trình*.

* **Kỹ thuật:** Sử dụng các thư viện như PyMuPDF hoặc Unstructured để trích xuất văn bản nhưng giữ lại cấu trúc tiêu đề. Mỗi "chunk" nên tương ứng với một quy trình nghiệp vụ hoàn chỉnh (ví dụ: "Quy trình xử lý khiếu nại đổi trả").18  
* **Lý do:** Nếu cắt đôi một quy trình, mô hình Teacher sẽ không có đủ ngữ cảnh để sinh ra một chuỗi suy luận logic đầy đủ, dẫn đến dữ liệu huấn luyện bị gãy vụn (hallucinated reasoning).20

#### **Giai đoạn 2: Tổng hợp Dữ liệu Suy luận (Synthetic Reasoning Generation)**

Đây là trái tim của Data Factory. Chúng ta sử dụng một mô hình mạnh (như GPT-4o, Claude 3.5 Sonnet hoặc DeepSeek-V3) để đóng vai trò "Giáo viên", chuyển đổi văn bản SOP thành các đoạn hội thoại có kèm suy luận.

Kỹ thuật Prompting cho Teacher Model:  
Prompt gửi cho Teacher Model cần được thiết kế kỹ lưỡng để ép mô hình này "suy nghĩ ra ngoài" (externalize thoughts).

* *Yêu cầu 1:* "Đóng vai một chuyên gia đào tạo nhân sự. Đọc quy trình SOP dưới đây và tạo ra một tình huống khách hàng thực tế."  
* *Yêu cầu 2 (Tạo suy luận):* "Hãy viết ra một chuỗi suy nghĩ nội tâm (\<think\>) nơi nhân viên phân tích yêu cầu của khách, đối chiếu với các điều kiện trong SOP, xác định thông tin còn thiếu, và quyết định bước tiếp theo.".16  
* *Yêu cầu 3 (Ánh xạ công cụ):* "Dựa trên kế hoạch, hãy tạo ra lệnh gọi công cụ (JSON) chính xác theo schema sau..."  
* *Yêu cầu 4 (Văn phong):* "Viết câu trả lời cuối cùng với giọng điệu chuyên nghiệp, đồng cảm nhưng quyết đoán, sử dụng đúng thuật ngữ nội bộ như 'Ticket', 'Escalation'...".22

#### **Giai đoạn 3: Kiểm định và Định dạng (Validation & Formatting)**

Dữ liệu sinh ra cần được kiểm tra tự động để loại bỏ các mẫu kém chất lượng.

* **Validation Script:** Sử dụng Python script để parse các lệnh JSON trong tool call. Nếu JSON không đúng cú pháp hoặc tham số không khớp với schema, mẫu dữ liệu đó sẽ bị loại bỏ hoặc đưa vào quy trình sửa lỗi tự động (Self-Correction Loop).14  
* **Unsloth Formatting:** Cuối cùng, dữ liệu sạch được chuyển đổi sang định dạng Chat Template mà Unsloth hỗ trợ (thường là định dạng Alpaca hoặc ShareGPT), đảm bảo các thẻ đặc biệt (special tokens) được xử lý đúng.18

## ---

**3\. Pipeline Fine-tuning Trên Unsloth: Triển Khai Kỹ Thuật**

Để đạt được mục tiêu tạo ra một agent vừa hiểu biết sâu (knowledgeable), vừa tư duy tốt (reasoning), vừa hành động chuẩn (actionable), và có văn phong thực (stylistic), chúng tôi đề xuất một chiến lược huấn luyện hai giai đoạn: **Supervised Fine-Tuning (SFT)** để nạp kiến thức và **Group Relative Policy Optimization (GRPO)** để rèn luyện tư duy và văn phong.

### **3.1 Thiết Lập Môi Trường Unsloth**

Việc thiết lập môi trường Unsloth đòi hỏi sự tương thích chặt chẽ giữa các thư viện để tận dụng tối đa khả năng tăng tốc phần cứng. Unsloth yêu cầu GPU NVIDIA (hỗ trợ tốt nhất từ dòng Ampere trở lên như A100, H100, hoặc dòng tiêu dùng RTX 3090/4090) và hệ điều hành Linux.3

Mã cài đặt cơ bản trong môi trường ảo:

Bash

conda create \--name unsloth\_env python=3.10  
conda activate unsloth\_env  
pip install unsloth vllm  
pip install \--no-deps "xformers\<0.0.27" "trl\<0.9.0" peft accelerate bitsandbytes

*Lưu ý:* Việc cài đặt vllm cùng với unsloth là bắt buộc cho giai đoạn GRPO để tăng tốc quá trình sinh dữ liệu (rollout generation).25

### **3.2 Giai đoạn 1: Supervised Fine-Tuning (SFT) \- Nạp Kiến Thức và Cấu Trúc**

Mục tiêu của giai đoạn này là dạy cho mô hình "biết" quy trình SOP, "biết" cách gọi tool, và "biết" cấu trúc \<think\>.

#### **Khởi tạo Mô hình**

Sử dụng FastLanguageModel để tải mô hình cơ sở (ví dụ: Llama-3.1-8B-Instruct) ở chế độ 4-bit quantization. Chế độ này giảm tải bộ nhớ VRAM xuống 4 lần mà gần như không làm giảm độ chính xác, cho phép huấn luyện các batch size lớn hơn.13

Python

from unsloth import FastLanguageModel  
import torch

model, tokenizer \= FastLanguageModel.from\_pretrained(  
    model\_name \= "unsloth/Llama-3.1-8B-Instruct",  
    max\_seq\_length \= 8192, \# Cần context dài cho chuỗi suy luận  
    dtype \= None, \# Tự động phát hiện (thường là bfloat16)  
    load\_in\_4bit \= True,  
)

#### **Cấu hình LoRA (Low-Rank Adaptation)**

Để mô hình có khả năng suy luận phức tạp, việc áp dụng LoRA lên tất cả các module tuyến tính (linear modules) là rất quan trọng, thay vì chỉ áp dụng lên các lớp Attention như truyền thống.

Python

model \= FastLanguageModel.get\_peft\_model(  
    model,  
    r \= 64, \# Rank cao hơn (64-128) giúp học logic phức tạp tốt hơn  
    target\_modules \= \["q\_proj", "k\_proj", "v\_proj", "o\_proj",  
                      "gate\_proj", "up\_proj", "down\_proj"\],  
    lora\_alpha \= 16,  
    lora\_dropout \= 0,  
    bias \= "none",  
    use\_gradient\_checkpointing \= "unsloth",   
    random\_state \= 3407,  
)

.3

#### **Cấu hình Trainer và Chat Template**

Điểm quan trọng nhất ở đây là sử dụng DataCollatorForCompletionOnlyLM. Kỹ thuật này đảm bảo rằng loss (hàm mất mát) chỉ được tính trên phần phản hồi của agent (bao gồm suy nghĩ và tool call), chứ không tính trên phần system prompt hay câu hỏi của người dùng. Điều này giúp mô hình tập trung tối đa vào việc học cách *xử lý* vấn đề.28

### **3.3 Giai đoạn 2: Group Relative Policy Optimization (GRPO) \- Rèn Luyện Tư Duy và Văn Phong**

Sau SFT, mô hình đã biết cách gọi tool nhưng có thể vẫn còn "ảo giác" (hallucination) trong suy luận hoặc văn phong chưa thật sự tự nhiên. GRPO là bước đột phá để giải quyết vấn đề này.

#### **Cơ chế hoạt động của GRPO**

Khác với PPO cần một mô hình Critic (tốn gấp đôi VRAM), GRPO hoạt động bằng cách sinh ra một nhóm (group) các câu trả lời (ví dụ: 8 câu trả lời) cho cùng một câu hỏi. Sau đó, nó chấm điểm các câu trả lời này dựa trên một tập hợp các hàm phần thưởng (reward functions) và cập nhật trọng số để ưu tiên các câu trả lời có điểm cao hơn trung bình của nhóm.29

#### **Thiết kế Hàm Phần Thưởng (Reward Functions)**

Đây là nơi chúng ta lập trình hóa các yêu cầu về "văn phong giống người thật" và "quy trình chuẩn". Chúng ta cần xây dựng 3 loại hàm phần thưởng:

1. **Phần thưởng Cú pháp (Format Reward):** Kiểm tra xem mô hình có tuân thủ cấu trúc \<think\>...\</think\> và định dạng JSON của tool call hay không. Đây là điều kiện tiên quyết (Hard constraint).31  
   * *Cài đặt:* Sử dụng Regex để parse output. Trả về điểm 1.0 nếu đúng cú pháp, 0.0 nếu sai.  
2. **Phần thưởng Chính xác (Correctness Reward):** Kiểm tra xem tool call được gọi có đúng với quy trình SOP không.  
   * *Cài đặt:* So sánh tên hàm và tham số trong tool call do mô hình sinh ra với tool call "Ground Truth" trong tập dữ liệu tổng hợp.  
3. **Phần thưởng Văn phong (Style/Persona Reward):** Đây là yếu tố giúp agent "giống người thật".  
   * *Cài đặt:* Chúng ta có thể sử dụng một mô hình embedding nhỏ hoặc các độ đo ngôn ngữ để so sánh văn bản phản hồi của agent với các mẫu văn bản chuẩn của nhân viên xuất sắc. Nếu độ tương đồng cosine cao, mô hình nhận điểm thưởng. Hoặc đơn giản hơn, phạt điểm nếu mô hình sử dụng các cụm từ "robot" như "I am an AI model".22

#### **Triển khai GRPO Trainer trên Unsloth**

Unsloth hỗ trợ trực tiếp GRPOTrainer từ thư viện TRL nhưng đã được tối ưu hóa hạt nhân.

Python

from trl import GRPOConfig, GRPOTrainer

training\_args \= GRPOConfig(  
    output\_dir \= "grpo\_agent\_output",  
    learning\_rate \= 1e-6, \# Learning rate rất thấp cho RL  
    num\_generations \= 8,  \# Kích thước nhóm (Group Size)  
    max\_completion\_length \= 1024, \# Dành không gian cho suy luận  
    beta \= 0.1, \# Hệ số phạt KL Divergence  
    use\_vllm \= True, \# Tích hợp vLLM để tăng tốc sinh dữ liệu  
    vllm\_gpu\_memory\_utilization \= 0.5, \# Chia sẻ VRAM  
)

trainer \= GRPOTrainer(  
    model \= model,  
    reward\_funcs \= \[format\_reward\_func, tool\_accuracy\_func, style\_reward\_func\],  
    args \= training\_args,  
    train\_dataset \= dataset,  
)  
trainer.train()

.25

Sự kết hợp giữa use\_vllm=True và cơ chế quản lý bộ nhớ của Unsloth cho phép quá trình sinh dữ liệu (rollout) diễn ra cực nhanh ngay trên cùng một GPU đang huấn luyện, điều mà các thư viện khác thường gặp khó khăn do nghẽn cổ chai bộ nhớ.3

## ---

**4\. Tổng Quan Về Dataset và Chiến Lược Xây Dựng**

Để mô hình có thể "dưa ra planning xử lý như một nhân viên chính thức", tập dữ liệu không thể chỉ là các cặp Hỏi-Đáp đơn giản. Nó phải là tập hợp của các "kịch bản xử lý" (processing scenarios).

### **4.1 Chi Tiết Về Record Dữ Liệu**

Mỗi bản ghi trong dataset (dòng trong file JSONL) đại diện cho một phiên làm việc xử lý sự cố.

JSON

{  
  "messages":  
}

*Lưu ý:* Trong ví dụ trên, phần \<think\> thể hiện rõ logic của một nhân viên đang tuân thủ quy trình. Phần \<tool\_call\> là hành động cụ thể. Nếu không có phần \<think\>, mô hình có thể sẽ "đoán mò" và thực hiện hoàn tiền ngay lập tức, vi phạm quy chế công ty.15

### **4.2 Tự Động Hóa Việc Xây Dựng Dataset**

Để tạo ra hàng nghìn mẫu dữ liệu như trên từ hàng trăm trang tài liệu PDF, chúng ta cần một pipeline tự động hóa:

1. **Trích xuất đặc trưng (Feature Extraction):** Phân tích PDF để tách riêng phần "Điều kiện" (Conditions) và "Hành động" (Actions). Ví dụ: "Nếu \[Điều kiện A\] thì thực hiện".  
2. **Kịch bản hóa (Scenario Generation):** Sử dụng LLM Teacher để tạo ra các kịch bản người dùng rơi vào các nhánh điều kiện khác nhau (ví dụ: Kịch bản người dùng thỏa mãn điều kiện A, và kịch bản người dùng vi phạm điều kiện A).  
3. **Làm giàu văn phong (Style Injection):** Yêu cầu LLM Teacher viết lại câu trả lời cuối cùng dựa trên bộ "Style Guide" của công ty (ví dụ: "Luôn bắt đầu bằng lời xin lỗi nếu từ chối", "Sử dụng ngôn ngữ tích cực").  
4. **Kiểm tra tính nhất quán (Consistency Check):** Một script tự động sẽ chạy lại logic của mẫu dữ liệu. Nếu phần suy luận \<think\> dẫn đến kết luận A, nhưng tool call lại thực hiện hành động B, mẫu dữ liệu đó sẽ bị loại bỏ.16

## ---

**5\. Triển Khai Thực Tế và Tương Lai**

### **5.1 Kiến Trúc Suy Luận (Inference Architecture)**

Sau khi tinh chỉnh thành công, mô hình được hợp nhất (merge) các trọng số LoRA vào mô hình gốc. Quá trình triển khai thực tế (Production) cần một kiến trúc suy luận đặc biệt để xử lý chuỗi tư duy:

* **Ẩn luồng tư duy:** Trong giao diện người dùng (UI Chatbot), phần nội dung trong thẻ \<think\>...\</think\> cần được ẩn đi. Người dùng cuối chỉ nhìn thấy câu trả lời tự nhiên hoặc kết quả thực thi công cụ. Tuy nhiên, luồng tư duy này cần được lưu log để đội ngũ kỹ thuật giám sát và debug (Audit Trail).8  
* **Vòng lặp thực thi (Execution Loop):** Hệ thống backend cần lắng nghe token kết thúc suy luận. Khi gặp thẻ \<tool\_call\>, hệ thống tạm dừng sinh văn bản, thực thi API thực tế, và đưa kết quả (JSON output) ngược trở lại vào ngữ cảnh chat để mô hình tiếp tục sinh ra câu trả lời cuối cùng cho người dùng.

### **5.2 Kết Luận**

Việc xây dựng Chatbot L1 với khả năng tư duy như nhân viên thực thụ không còn là viễn cảnh xa vời nhờ sự kết hợp giữa phương pháp Agentic Fine-tuning và sức mạnh tối ưu hóa của Unsloth. Pipeline được đề xuất trong báo cáo này—từ việc xây dựng Data Factory đến quy trình huấn luyện kép SFT+GRPO—cung cấp một lộ trình rõ ràng, khả thi về mặt kỹ thuật và hiệu quả về mặt chi phí cho các doanh nghiệp muốn sở hữu hệ thống AI tự chủ cao cấp.

Bằng cách nhúng sâu quy trình (SOP) và văn phong (Persona) vào trọng số mô hình, doanh nghiệp không chỉ giảm thiểu rủi ro ảo giác mà còn tạo ra trải nghiệm khách hàng đồng nhất, chuyên nghiệp, đánh dấu sự trưởng thành thực sự của AI trong môi trường doanh nghiệp vào năm 2025\.

### **Tài liệu tham khảo & Nguồn dữ liệu**

* **Xu hướng & Chiến lược:** 1  
* **Kỹ thuật Unsloth & Tối ưu hóa:** 10  
* **Thuật toán GRPO & RL:** 12  
* **Xây dựng Dataset & Dữ liệu tổng hợp:** 14  
* **Văn phong & Persona:** 9  
* **Sử dụng công cụ & Suy luận:** 8

#### **Works cited**

1. Top 10 trends in AI adoption for enterprises in 2025 \- Glean, accessed December 29, 2025, [https://www.glean.com/perspectives/enterprise-insights-from-ai](https://www.glean.com/perspectives/enterprise-insights-from-ai)  
2. What's next for AI? \- Deloitte, accessed December 29, 2025, [https://www.deloitte.com/us/en/insights/topics/technology-management/tech-trends/2025/tech-trends-ai-agents-and-autonomous-ai.html](https://www.deloitte.com/us/en/insights/topics/technology-management/tech-trends/2025/tech-trends-ai-agents-and-autonomous-ai.html)  
3. unslothai/unsloth: Fine-tuning & Reinforcement Learning for LLMs. 🦥 Train OpenAI gpt-oss, DeepSeek-R1, Qwen3, Gemma 3, TTS 2x faster with 70% less VRAM. \- GitHub, accessed December 29, 2025, [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)  
4. RAG vs Fine-Tuning 2026 What You Need to Know Before Implementation \- Kanerika, accessed December 29, 2025, [https://kanerika.com/blogs/rag-vs-fine-tuning/](https://kanerika.com/blogs/rag-vs-fine-tuning/)  
5. RAG vs fine tuning for help centers: The 2025 guide \- eesel AI, accessed December 29, 2025, [https://www.eesel.ai/blog/rag-vs-fine-tuning-for-help-centers](https://www.eesel.ai/blog/rag-vs-fine-tuning-for-help-centers)  
6. RAG vs. Fine-tuning \- IBM, accessed December 29, 2025, [https://www.ibm.com/think/topics/rag-vs-fine-tuning](https://www.ibm.com/think/topics/rag-vs-fine-tuning)  
7. Advanced Techniques in Agent Fine-Tuning for 2025 \- Sparkco, accessed December 29, 2025, [https://sparkco.ai/blog/advanced-techniques-in-agent-fine-tuning-for-2025](https://sparkco.ai/blog/advanced-techniques-in-agent-fine-tuning-for-2025)  
8. DeepSeek R1 Quickstart \- Together.ai Docs, accessed December 29, 2025, [https://docs.together.ai/docs/deepseek-r1](https://docs.together.ai/docs/deepseek-r1)  
9. LLM Fine‑Tuning in 2025: A Hands‑On, Test‑Driven Blueprint | by Carlos Esteban | Medium, accessed December 29, 2025, [https://medium.com/@tabers77/llm-fine-tuning-in-2025-a-hands-on-test-driven-blueprint-dd1c7887bb99](https://medium.com/@tabers77/llm-fine-tuning-in-2025-a-hands-on-test-driven-blueprint-dd1c7887bb99)  
10. Unsloth AI: A Deep Dive into Faster, More Efficient LLM Fine-Tuning \- Skywork.ai, accessed December 29, 2025, [https://skywork.ai/skypage/en/Unsloth-AI:-A-Deep-Dive-into-Faster,-More-Efficient-LLM-Fine-Tuning/1972856091659923456](https://skywork.ai/skypage/en/Unsloth-AI:-A-Deep-Dive-into-Faster,-More-Efficient-LLM-Fine-Tuning/1972856091659923456)  
11. Unsloth: Making LLM Fine-Tuning Fast, Cheap, and Practical | by Asimsultan (Head of AI) | Nov, 2025 | Medium, accessed December 29, 2025, [https://medium.com/@asimsultan2/unsloth-making-llm-fine-tuning-fast-cheap-and-practical-f324bcc98bd8](https://medium.com/@asimsultan2/unsloth-making-llm-fine-tuning-fast-cheap-and-practical-f324bcc98bd8)  
12. GRPO Fine-Tuning on DeepSeek-7B with Unsloth \- Analytics Vidhya, accessed December 29, 2025, [https://www.analyticsvidhya.com/blog/2025/02/grpo-fine-tuning-on-deepseek-7b/](https://www.analyticsvidhya.com/blog/2025/02/grpo-fine-tuning-on-deepseek-7b/)  
13. Fine-tuning Llama 3.2 and Using It Locally: A Step-by-Step Guide | DataCamp, accessed December 29, 2025, [https://www.datacamp.com/tutorial/fine-tuning-llama-3-2](https://www.datacamp.com/tutorial/fine-tuning-llama-3-2)  
14. Fine-Tuning LLMs for Efficient Agentic Tasks with Hyperstack AI Studio, accessed December 29, 2025, [https://www.hyperstack.cloud/technical-resources/tutorials/fine-tuning-llms-for-agentic-use-with-hyperstack-ai-studio](https://www.hyperstack.cloud/technical-resources/tutorials/fine-tuning-llms-for-agentic-use-with-hyperstack-ai-studio)  
15. chat\_template.jinja · unsloth/Llama-4-Scout-17B-16E-Instruct at main \- Hugging Face, accessed December 29, 2025, [https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/chat\_template.jinja](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.jinja)  
16. SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents \- arXiv, accessed December 29, 2025, [https://arxiv.org/html/2506.08119v1](https://arxiv.org/html/2506.08119v1)  
17. Fine-tuning LLMs for function-calling \- Wandb, accessed December 29, 2025, [https://wandb.ai/wandb/function-calling-finetuning/reports/Fine-tuning-LLMs-for-function-calling--VmlldzoxMjgxMTgxMg](https://wandb.ai/wandb/function-calling-finetuning/reports/Fine-tuning-LLMs-for-function-calling--VmlldzoxMjgxMTgxMg)  
18. Converting and Storing Text Chunks in JSONL Format | CodeSignal Learn, accessed December 29, 2025, [https://codesignal.com/learn/courses/chunking-and-storing-text-for-efficient-llm-processing/lessons/converting-and-storing-text-chunks-in-jsonl-format](https://codesignal.com/learn/courses/chunking-and-storing-text-for-efficient-llm-processing/lessons/converting-and-storing-text-chunks-in-jsonl-format)  
19. From PDFs to AI-ready structured data: a deep dive \- Explosion AI, accessed December 29, 2025, [https://explosion.ai/blog/pdfs-nlp-structured-data](https://explosion.ai/blog/pdfs-nlp-structured-data)  
20. Using LLMs for Synthetic Data Generation: The Definitive Guide \- Confident AI, accessed December 29, 2025, [https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms)  
21. Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use \- arXiv, accessed December 29, 2025, [https://arxiv.org/html/2504.04736v1](https://arxiv.org/html/2504.04736v1)  
22. GRPO \- Reward functions for medical reasoning : r/unsloth \- Reddit, accessed December 29, 2025, [https://www.reddit.com/r/unsloth/comments/1iw7675/grpo\_reward\_functions\_for\_medical\_reasoning/](https://www.reddit.com/r/unsloth/comments/1iw7675/grpo_reward_functions_for_medical_reasoning/)  
23. Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning, accessed December 29, 2025, [https://arxiv.org/html/2512.05747v1](https://arxiv.org/html/2512.05747v1)  
24. Fine-Tuning Made Fast : How Unsloth is Redefining the LLM Training Workflow \- Medium, accessed December 29, 2025, [https://medium.com/@mehtameet115/fine-tuning-made-fast-how-unsloth-is-redefining-the-llm-training-workflow-db511353957c](https://medium.com/@mehtameet115/fine-tuning-made-fast-how-unsloth-is-redefining-the-llm-training-workflow-db511353957c)  
25. GRPO Trainer \- Hugging Face, accessed December 29, 2025, [https://huggingface.co/docs/trl/main/en/grpo\_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)  
26. Fine-Tuning 1B LLaMA 3.2: A Comprehensive Step-by-Step Guide with Code, accessed December 29, 2025, [https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article](https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article)  
27. Finetuning gpt-oss-20b on custom tool calling. : r/unsloth \- Reddit, accessed December 29, 2025, [https://www.reddit.com/r/unsloth/comments/1oz9spx/finetuning\_gptoss20b\_on\_custom\_tool\_calling/](https://www.reddit.com/r/unsloth/comments/1oz9spx/finetuning_gptoss20b_on_custom_tool_calling/)  
28. 4\. Choosing the learning paradigm — From Text to Insight, accessed December 29, 2025, [https://matextract.pub/content/finetune/choosing\_paradigm.html](https://matextract.pub/content/finetune/choosing_paradigm.html)  
29. Deep dive into Group Relative Policy Optimization (GRPO) \- AWS Builder Center, accessed December 29, 2025, [https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo](https://builder.aws.com/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo)  
30. Fine-Tuning LLMs: A Look at Group Relative Policy Optimization (GRPO) \- Medium, accessed December 29, 2025, [https://medium.com/@g.anirudh15/fine-tuning-llms-a-look-at-group-relative-policy-optimization-grpo-8240cac48ebc](https://medium.com/@g.anirudh15/fine-tuning-llms-a-look-at-group-relative-policy-optimization-grpo-8240cac48ebc)  
31. What should I expect from GPRO / adding reasoning to base model? : r/unsloth \- Reddit, accessed December 29, 2025, [https://www.reddit.com/r/unsloth/comments/1jcnx0b/what\_should\_i\_expect\_from\_gpro\_adding\_reasoning/](https://www.reddit.com/r/unsloth/comments/1jcnx0b/what_should_i_expect_from_gpro_adding_reasoning/)  
32. Train an LLM on NVIDIA Blackwell with Unsloth—and Scale for Production, accessed December 29, 2025, [https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/)  
33. Generating Synthetic Datasets for LLM Evaluators & Agents \- Phoenix \- Arize AI, accessed December 29, 2025, [https://arize.com/docs/phoenix/cookbook/tracing/generating-synthetic-datasets-for-llm-evaluators-and-agents](https://arize.com/docs/phoenix/cookbook/tracing/generating-synthetic-datasets-for-llm-evaluators-and-agents)  
34. Llama 4 Overpromises but Underdelivers \- unwind ai, accessed December 29, 2025, [https://www.theunwindai.com/p/llama-4-overpromises-but-underdelivers](https://www.theunwindai.com/p/llama-4-overpromises-but-underdelivers)  
35. \[2510.08191\] Training-Free Group Relative Policy Optimization \- arXiv, accessed December 29, 2025, [https://arxiv.org/abs/2510.08191](https://arxiv.org/abs/2510.08191)  
36. Flow-of-Action: SOP Enhanced LLM-Based Multi-Agent System for Root Cause Analysis, accessed December 29, 2025, [https://arxiv.org/html/2502.08224v1](https://arxiv.org/html/2502.08224v1)  
37. GRPO and the Future of LLM Fine-tuning: Moving Beyond Human Imitation \- Medium, accessed December 29, 2025, [https://medium.com/@andrecnf/grpo-and-the-future-of-llm-fine-tuning-moving-beyond-human-imitation-335dc14c2df9](https://medium.com/@andrecnf/grpo-and-the-future-of-llm-fine-tuning-moving-beyond-human-imitation-335dc14c2df9)
from src.data_factory.reconstruction import reconstruct_dataset
import asyncio


if __name__ == "__main__":
    asyncio.run(reconstruct_dataset(
        input_path="data/raw/conversations.jsonl",
        output_path="data/processed/annotated_conversations.jsonl",
        model_name="gpt-5.2",
        max_concurrent=4  # Process 3 conversations in parallel
    ))
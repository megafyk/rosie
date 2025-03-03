import json
import os
import time
from pathlib import Path

from config import LLM_MODEL_NAME, TEST_PART_SYSTEM
from generate import generate
from ragas import EvaluationDataset, evaluate
from ragas.metrics import BleuScore, RougeScore
from search import search


# Function to save intermediate results
def save_intermediate_results(results, file_path, idx):
    try:
        temp_path = f"{file_path}.temp_{idx}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        # Safely replace the file
        if os.path.exists(file_path):
            os.replace(temp_path, file_path)
        else:
            os.rename(temp_path, file_path)
        print(f"Saved intermediate results after processing item {idx}")
    except Exception as e:
        print(f"Error saving intermediate results: {e}")


if __name__ == "__main__":
    try:
        # Use pathlib for more robust path handling
        script_dir = Path(__file__).parent
        eval_file_path = script_dir.parent / "datasets" / "eval.json"
        result_file_path = script_dir.parent / "datasets" / "eval_result.json"

        # Check if we can resume from previous run
        eval_res = []
        if os.path.exists(result_file_path):
            try:
                with open(result_file_path, encoding="utf-8") as f:
                    eval_res = json.load(f)
                print(f"Resuming from existing results with {len(eval_res)} items")
            except Exception as e:
                print(f"Error loading existing results: {e}")

        # Load evaluation data
        with open(eval_file_path, encoding="utf-8") as f:
            eval_data = json.load(f)

        total_questions = len(eval_data)
        processed_questions = len(eval_res)

        # Process only remaining questions
        for idx in range(processed_questions, total_questions):
            q = eval_data[idx]
            question = q["question"]
            expected = q["answer"]
            print(f"Processing question {idx + 1}/{total_questions}: {question}...")

            try:
                # Add delays to avoid rate limits
                retrieved_contexts = search(question)
                time.sleep(1)  # Small delay

                messages = [
                    {"role": "system", "content": TEST_PART_SYSTEM},
                    {"role": "assistant", "content": retrieved_contexts},
                    {"role": "user", "content": question},
                ]

                print(f"Local RAG generate query {idx + 1}/{total_questions}...")
                local = generate(
                    f"local:{LLM_MODEL_NAME}",
                    messages,
                    {
                        "temperature": 0.2,
                        "max_new_tokens": 128,
                        "return_full_text": False,
                        "device": "cpu",
                    },
                )

                # Ensure local_response is string
                local_response = (
                    local.choices[0].message.content
                    if hasattr(local, "choices") and len(local.choices) > 0
                    else str(local)
                )

                eval_datasets = EvaluationDataset.from_list(
                    [
                        {
                            "user_input": question,
                            "retrieved_contexts": [retrieved_contexts],
                            "response": local_response,
                            "reference": expected,
                        }
                    ]
                )

                evaluation = evaluate(
                    eval_datasets, metrics=[BleuScore(), RougeScore()]
                )

                time.sleep(1)  # Small delay

                # Add result and save intermediate results
                eval_res.append(
                    {
                        "question": question,
                        "evaluation": evaluation.to_pandas().to_dict(orient="records"),
                    }
                )

                # Save after each item
                save_intermediate_results(eval_res, result_file_path, idx)
                print(f"Completed evaluation {idx + 1}/{total_questions}")

                # Add a delay between iterations to give system time to recover resources
                time.sleep(2)

            except Exception as e:
                print(f"Error processing question {idx + 1}: {str(e)}")
                # Add the failed question to results with error information
                eval_res.append(
                    {
                        "question": question,
                        "local": "",
                        "reference": "",
                        "evaluation": f"ERROR: {str(e)}",
                    }
                )
                # Save even on error
                save_intermediate_results(eval_res, result_file_path, idx)
                time.sleep(5)  # Longer delay after error
        print(f"Evaluation complete. Results saved to {result_file_path}")
    except Exception as e:
        print(f"Fatal error: {str(e)}")

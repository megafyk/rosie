import json
import os
import time
from pathlib import Path

from config import EVAL_SYSTEM, LLM_MODEL_NAME, OPENROUTER_MODEL, TEST_PART_SYSTEM
from generate import generate, generate_openrouter
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
            question = q['question']
            print(f"Processing question {idx+1}/{total_questions}: {question}...")

            try:
                # Add delays to avoid rate limits
                context = search(question)
                time.sleep(1)  # Small delay

                messages = [
                    {"role": "system", "content": TEST_PART_SYSTEM},
                    {"role": "assistant", "content": context},
                    {"role": "user", "content": question}
                ]

                print(f"Local RAG generate query {idx+1}/{total_questions}...")
                local_response = generate(f"local:{LLM_MODEL_NAME}", messages, {
                    "temperature": 0.2,
                    "max_new_tokens": 128,
                    "return_full_text": False,
                    "device": "cpu"
                })
                print(local_response.choices[0].message.content)
                # Ensure local_response is string
                local = local_response.choices[0].message.content if hasattr(local_response, 'choices') and len(local_response.choices) > 0 else str(local_response)
                time.sleep(1)  # Small delay


                print(f"Reference model generate {idx+1}/{total_questions}...")
                # refer_response = generate_openrouter(OPENROUTER_MODEL, messages, {
                #     "temperature": 0.2,
                #     "max_completion_tokens": 128
                # })

                refer_response = generate(f"local:{LLM_MODEL_NAME}", messages, {
                    "temperature": 0.2,
                    "max_new_tokens": 128,
                    "return_full_text": False,
                    "device": "cpu"
                })
                print(refer_response.choices[0].message.content)
                # Extract content safely
                refer = refer_response.choices[0].message.content if hasattr(refer_response, 'choices') and len(refer_response.choices) > 0 else str(refer_response)
                time.sleep(2)  # Slightly longer delay for API rate limits

                print(f"Evaluating RAG {idx+1}/{total_questions}...")
                eval_messages = [
                    {"role": "system", "content": EVAL_SYSTEM},
                    {"role": "assistant", "content": f"generated_answer: {local}"},
                    {"role": "user", "content": f"query: {question}\nreference_answer: {refer}"}
                ]


                # evaluate_response = generate_openrouter(OPENROUTER_MODEL, eval_messages, {
                #     "temperature": 0.1,
                #     "max_completion_tokens": 128
                # })

                evaluate_response = generate(f"local:{LLM_MODEL_NAME}", eval_messages, {
                    "temperature": 0.1,
                    "max_new_tokens": 128,
                    "return_full_text": False,
                    "device": "cpu"
                })
                print(evaluate_response.choices[0].message.content)

                # Safely extract content
                evaluation = evaluate_response.choices[0].message.content if hasattr(evaluate_response, 'choices') and len(evaluate_response.choices) > 0 else str(evaluate_response)

                # Add result and save intermediate results
                eval_res.append({
                    "question": question,
                    "local": local,
                    "reference": refer,
                    "evaluation": evaluation
                })

                # Save after each item
                save_intermediate_results(eval_res, result_file_path, idx)
                print(f"Completed evaluation {idx+1}/{total_questions}")

                # Add a delay between iterations to give system time to recover resources
                time.sleep(2)

            except Exception as e:
                print(f"Error processing question {idx+1}: {str(e)}")
                # Add the failed question to results with error information
                eval_res.append({
                    "question": question,
                    "local": "",
                    "reference": "",
                    "evaluation": f"ERROR: {str(e)}"
                })
                # Save even on error
                save_intermediate_results(eval_res, result_file_path, idx)
                time.sleep(5)  # Longer delay after error
        print(f"Evaluation complete. Results saved to {result_file_path}")
    except Exception as e:
        print(f"Fatal error: {str(e)}")

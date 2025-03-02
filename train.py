import faiss
import numpy as np
from vllm import LLM
from sentence_transformers import SentenceTransformer
import json
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler

# Initialize model, tokenizer, memory
model = LLM("agentica/deepscaler-1.5b-preview")
tokenizer = AutoTokenizer.from_pretrained("agentica/deepscaler-1.5b-preview")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
memory_bank = faiss.IndexFlatL2(dimension)
memory_texts = []

# Lengths
X_input = 6500
X_output = 7000
partitions_input = {
    "problem": (0, 1000),
    "thought": (1000, 5000),  # 4K
    "memory_query": (5000, 5500),  # 500
    "memory_answer": (5500, 6500),  # 1K
}
partitions_output = {
    "thought": (1000, 5000),  # 4K
    "memory_query": (5000, 5500),  # 500
    "write_to_memory": (6500, 7500),  # 1K
    "answer": (7500, 8000),  # 500
}

# Load GSM8K dataset (subset: first 100 problems)
with open("gsm8k_train.json", "r") as f:
    dataset = json.load(f)[:100]  # Small subset

# PPO Config (GRPO-like)
ppo_config = PPOConfig(
    model_name="agentica/deepscaler-1.5b-preview",
    learning_rate=1e-5,
    ppo_epochs=4,
    mini_batch_size=4,
    batch_size=16,
    clip_eps=0.2,  # Clipping for stability
    kl_penalty="kl",  # GRPO typically uses KL penalty
    target_kl=0.01,  # Target KL divergence
)

# PPO Trainer (mock GRPO by customizing PPO)
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,  # Could use original model as reference
    tokenizer=tokenizer,
)


# Helpers
def extract_tag(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.find(start_tag) + len(start_tag)
    end_idx = text.find(end_tag)
    if start_idx >= len(start_tag) and end_idx > start_idx:
        return text[start_idx:end_idx].strip()
    return ""


def add_to_memory(text, tag):
    embedding = embedding_model.encode([text])[0]
    memory_bank.add(np.array([embedding]))
    memory_texts.append({"text": text, "tag": tag})


def search_memory(query, max_tokens=1000):
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = memory_bank.search(np.array([query_embedding]), k=1)
    if indices[0][0] >= 0:
        memory = memory_texts[indices[0][0]]["text"]
        return memory[:max_tokens]
    return "No memory queried"


# Seed memory
add_to_memory("Tutorial: Break into steps, query memory, write insights", "tutorial")

# Training loop
num_episodes = len(dataset)
output_length_sampler = LengthSampler(X_output, X_output + 1)  # Fixed output length
for episode in range(num_episodes):
    problem_data = dataset[episode]
    problem_text = f"Problem: {problem_data['question']}"
    correct_answer = problem_data["answer"]
    problem = problem_text + " " * (1000 - len(problem_text))

    # Initial prompt
    prompt = (
        problem
        + "<thought></thought>"
        + " " * (5000 - 1000 - len("<thought></thought>"))
        + "<memory_query></memory_query>"
        + " " * (5500 - 5000 - len("<memory_query></memory_query>"))
        + "<memory_answer></memory_answer>"
        + " " * (6500 - 5500 - len("<memory_answer></memory_answer>"))
    )

    # Add problem to memory
    add_to_memory(problem_text, "current_problem")

    # Solve loop
    max_steps = 10
    format_penalties = []
    step_rewards = []
    for step in range(max_steps):
        # Step 1: Encode prompt and generate
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
            torch.device("cuda")
        )
        outputs = model.generate(
            input_ids=input_ids, max_tokens=X_output, temperature=0, num_beams=1
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Step 2: Post-NN-Call Processing
        sections = {
            "thought": extract_tag(output, "thought"),
            "memory_query": extract_tag(output, "memory_query"),
            "write_to_memory": extract_tag(output, "write_to_memory"),
            "answer": extract_tag(output, "answer"),
        }

        # Check format compliance
        format_penalty = 0
        for tag, content in sections.items():
            start, end = partitions_output[tag]
            if len(content) > (end - start):
                format_penalty -= 0.5
        format_penalties.append(format_penalty)

        # Memory query
        memory_query = sections["memory_query"]
        if memory_query.strip():
            memory_answer = search_memory(memory_query)
        else:
            memory_answer = "No memory queried"

        # Write to memory
        write_content = sections["write_to_memory"]
        if write_content.strip():
            add_to_memory(write_content, f"step_{step}_ep_{episode}")

        # Check answer
        answer = sections["answer"]
        reward = 0
        if answer.strip():
            is_correct = str(answer) == str(correct_answer)
            reward = 1 if is_correct else 0
            print(
                f"Episode {episode + 1}, Step {step + 1}: Answer: {answer}, Correct: {is_correct} (Expected: {correct_answer})"
            )
            if is_correct:
                print(f"Solved in {step + 1} steps!")
                break
        reward += format_penalty - 0.1  # Per-step penalty
        step_rewards.append(reward)

        # Update prompt
        prompt = (
            problem
            + f"<thought>{sections['thought']}</thought>"
            + " " * (5000 - 1000 - len(f"<thought>{sections['thought']}</thought>"))
            + f"<memory_query>{sections['memory_query']}</memory_query>"
            + " "
            * (
                5500
                - 5000
                - len(f"<memory_query>{sections['memory_query']}</memory_query>")
            )
            + f"<memory_answer>{memory_answer}</memory_answer>"
            + " "
            * (6500 - 5500 - len(f"<memory_answer>{memory_answer}</memory_answer>"))
        )

    # Final reward for episode
    total_reward = sum(step_rewards)
    episode_rollout = {
        "query": tokenizer.encode(prompt, return_tensors="pt"),
        "response": output,
        "reward": total_reward,
    }

    # GRPO training (batch rollouts)
    ppo_trainer.step(
        query=episode_rollout["query"],
        response=tokenizer.encode(output, return_tensors="pt"),
        reward=torch.tensor([total_reward]),
    )
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Final log
print("Training completed")

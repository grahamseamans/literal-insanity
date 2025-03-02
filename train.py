import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import json
import torch.nn.functional as F

# Initialize model, tokenizer, memory
model = AutoModelForCausalLM.from_pretrained(
    "agentica-org/DeepScaleR-1.5B-Preview"
).cuda()
tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
memory_bank = faiss.IndexFlatL2(dimension)
memory_texts = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # Reduced learning rate

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
    dataset = json.load(f)[:100]


# Helpers
def extract_tag(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.find(start_tag) + len(start_tag)
    end_idx = text.find(end_tag)
    if start_idx >= len(start_tag) and end_idx > start_idx:
        return text[start_idx:end_idx].strip()
    return ""


# Memory management settings
MAX_MEMORY_SIZE = 100
MEMORY_PRUNE_THRESHOLD = 80


def add_to_memory(text, tag):
    # Check if memory needs pruning
    if len(memory_texts) >= MEMORY_PRUNE_THRESHOLD:
        # Remove oldest entries except tutorial
        to_remove = []
        for i, mem in enumerate(memory_texts):
            if mem["tag"] != "tutorial":
                to_remove.append(i)
                if len(memory_texts) - len(to_remove) <= MAX_MEMORY_SIZE:
                    break

        # Remove from FAISS index and memory_texts
        if to_remove:
            memory_bank.remove_ids(np.array(to_remove))
            for i in reversed(to_remove):
                memory_texts.pop(i)
            print(
                f"Pruned {len(to_remove)} memories. Current size: {len(memory_texts)}"
            )

    # Add new memory
    embedding = embedding_model.encode([text])[0]
    memory_bank.add(np.array([embedding]))
    memory_texts.append({"text": text, "tag": tag})


def search_memory(query, max_tokens=1000):
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = memory_bank.search(np.array([query_embedding]), k=1)
    if indices[0][0] >= 0:
        memory = memory_texts[indices[0][0]]["text"]
        print(f"Memory found (distance={distances[0][0]:.3f}): {memory[:50]}...")
        return memory[:max_tokens]
    print("No relevant memory found")
    return "No memory queried"


# Seed memory
add_to_memory("Tutorial: Break into steps, query memory, write insights", "tutorial")

# PPO settings
clip_eps = 0.2
batch_size = 2  # Reduced batch size
max_steps = 5  # Reduced max steps
max_grad_norm = 1.0  # For gradient clipping

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Training loop
num_episodes = len(dataset)
rollouts = []
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
    step_rewards = []
    for step in range(max_steps):
        # Forward pass with structured decoding
        torch.cuda.empty_cache()  # Clear cache before forward pass
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.amp.autocast(device_type="cuda"):  # Updated autocast syntax
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits[:, -X_output:]  # Predict X_output tokens
            probs = F.softmax(logits, dim=-1)
            output_ids = torch.argmax(probs, dim=-1)  # Greedy decode
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Compute log probabilities for PPO
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(
                dim=-1, index=output_ids.unsqueeze(-1)
            ).squeeze(-1)

        # Process output
        sections = {}
        current_pos = 0
        for tag, (start, end) in partitions_output.items():
            # Get logits for this section
            section_length = end - start
            section_text = output[current_pos : current_pos + section_length]
            sections[tag] = section_text.strip()
            current_pos += section_length

        # Format sections with tags
        output = (
            f"<thought>{sections['thought']}</thought>"
            f"<memory_query>{sections['memory_query']}</memory_query>"
            f"<write_to_memory>{sections['write_to_memory']}</write_to_memory>"
            f"<answer>{sections['answer']}</answer>"
        )

        # Compute format penalty based on token counts
        format_penalty = 0
        for tag, content in sections.items():
            start, end = partitions_output[tag]
            token_count = len(tokenizer.encode(content))
            if token_count > (end - start):
                format_penalty -= 1.0  # Stronger penalty for token overflow

        # Memory query with reward
        memory_query = sections["memory_query"]
        memory_reward = 0
        if memory_query.strip():
            print(f"\nStep {step + 1} Memory Query: {memory_query}")
            memory_answer = search_memory(memory_query)
            # Increased reward for memory usage
            if memory_answer != "No memory queried":
                memory_reward = 0.3  # Increased from 0.2
                print(f"Memory reward: {memory_reward}")
        else:
            memory_answer = "No memory queried"
            memory_reward = -0.1  # Penalty for not using memory
            print("No memory query made")

        # Write to memory with penalty for redundancy
        write_content = sections["write_to_memory"]
        memory_write_penalty = 0
        if write_content.strip():
            print(f"\nAttempting to write to memory: {write_content[:50]}...")
            # Check for redundant writes
            if write_content in [m["text"] for m in memory_texts]:
                memory_write_penalty = -0.2  # Increased penalty for redundancy
                print("Redundant memory write detected")
            else:
                memory_write_penalty = 0.1  # Reward for unique memory writes
            add_to_memory(write_content, f"step_{step}_ep_{episode}")
            print(f"Memory write penalty: {memory_write_penalty}")

        # Reward calculation
        answer = sections["answer"]
        reward = 0
        if answer.strip():
            is_correct = str(answer) == str(correct_answer)
            reward = 1 if is_correct else -0.2  # Added penalty for wrong answers
            print(
                f"Episode {episode + 1}, Step {step + 1}: Answer: {answer}, Correct: {is_correct}"
            )
            if is_correct:
                # Extra reward if memory helped
                if memory_answer != "No memory queried":
                    reward += 0.5  # Increased from 0.3
                break

        # Combine all rewards
        reward += (
            format_penalty  # Format compliance
            + memory_reward  # Memory query reward
            + memory_write_penalty  # Memory write penalty
            - 0.1  # Step penalty
        )
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

    # Collect rollout
    total_reward = sum(step_rewards)
    query_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    response_ids = output_ids.clone()  # Use the generated output IDs

    # Store log probabilities instead of logits
    rollouts.append((query_ids, response_ids, chosen_log_probs, total_reward))
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # PPO training every batch_size episodes
    if (episode + 1) % batch_size == 0:
        # Unpack rollouts
        queries, responses, old_log_probs, rewards = zip(*rollouts)

        # Find max lengths
        max_query_len = max(q.size(1) for q in queries)
        max_response_len = max(r.size(1) for r in responses)

        # Pad sequences
        padded_queries = torch.zeros(
            len(queries), max_query_len, dtype=torch.long, device="cuda"
        )
        padded_responses = torch.zeros(
            len(responses), max_response_len, dtype=torch.long, device="cuda"
        )
        padded_old_log_probs = torch.zeros(
            len(old_log_probs), max_response_len, device="cuda"
        )

        # Pad sequences
        for i, (q, r, olp) in enumerate(zip(queries, responses, old_log_probs)):
            padded_queries[i, : q.size(1)] = q[0]
            padded_responses[i, : r.size(1)] = r[0]
            padded_old_log_probs[i, : olp.size(1)] = olp[0]

        rewards = torch.tensor(rewards, device="cuda")

        # Forward pass for new log probs
        torch.cuda.empty_cache()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(padded_queries)
            new_logits = outputs.logits[:, -X_output:]
            new_log_probs = (
                F.log_softmax(new_logits, dim=-1)
                .gather(dim=-1, index=padded_responses.unsqueeze(-1))
                .squeeze(-1)
            )

        # Debug info
        print(
            f"New log probs: {new_log_probs.mean().item()}, Old log probs: {padded_old_log_probs.mean().item()}"
        )

        # Compute mask for valid tokens
        response_mask = (padded_responses != 0).float()

        # Compute PPO loss
        advantages = rewards.unsqueeze(-1).expand_as(new_log_probs) - rewards.mean()
        ratio = torch.exp(new_log_probs - padded_old_log_probs)

        # Debug ratio statistics
        print(
            f"Ratio mean: {ratio.mean().item()}, min: {ratio.min().item()}, max: {ratio.max().item()}"
        )

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2) * response_mask
        policy_loss = (
            policy_loss.sum() / response_mask.sum()
        )  # Average over valid tokens

        # Update with gradient clipping
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        rollouts = []  # Clear batch
        print(f"Batch {episode // batch_size + 1}: Policy Loss = {policy_loss.item()}")

# Final training on remaining rollouts
if rollouts:
    # Unpack rollouts
    queries, responses, old_log_probs, rewards = zip(*rollouts)

    # Find max lengths
    max_query_len = max(q.size(1) for q in queries)
    max_response_len = max(r.size(1) for r in responses)

    # Pad sequences
    padded_queries = torch.zeros(
        len(queries), max_query_len, dtype=torch.long, device="cuda"
    )
    padded_responses = torch.zeros(
        len(responses), max_response_len, dtype=torch.long, device="cuda"
    )
    padded_old_log_probs = torch.zeros(
        len(old_log_probs), max_response_len, device="cuda"
    )

    for i, (q, r, olp) in enumerate(zip(queries, responses, old_log_probs)):
        padded_queries[i, : q.size(1)] = q[0]
        padded_responses[i, : r.size(1)] = r[0]
        padded_old_log_probs[i, : olp.size(1)] = olp[0]

    rewards = torch.tensor(rewards, device="cuda")

    # Forward pass for new log probs
    torch.cuda.empty_cache()
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(padded_queries)
        new_logits = outputs.logits[:, -X_output:]
        new_log_probs = (
            F.log_softmax(new_logits, dim=-1)
            .gather(dim=-1, index=padded_responses.unsqueeze(-1))
            .squeeze(-1)
        )

    # Debug info
    print(
        f"Final batch - New log probs: {new_log_probs.mean().item()}, Old log probs: {padded_old_log_probs.mean().item()}"
    )

    # Compute mask for valid tokens
    response_mask = (padded_responses != 0).float()

    # Compute PPO loss
    advantages = rewards.unsqueeze(-1).expand_as(new_log_probs) - rewards.mean()
    ratio = torch.exp(new_log_probs - padded_old_log_probs)

    # Debug ratio statistics
    print(
        f"Final batch - Ratio mean: {ratio.mean().item()}, min: {ratio.min().item()}, max: {ratio.max().item()}"
    )

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2) * response_mask
    policy_loss = policy_loss.sum() / response_mask.sum()  # Average over valid tokens

    # Update with gradient clipping
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    print(f"Final Batch: Policy Loss = {policy_loss.item()}")

print("Training completed")

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import json
import torch.nn.functional as F
import time
from tqdm import tqdm

torch.cuda.empty_cache()
print("Starting script...")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize model with quantization config
print("Loading DeepScaleR-1.5B-Preview...")
start_time = time.time()
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "agentica-org/DeepScaleR-1.5B-Preview",
    quantization_config=quant_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")
print(f"GPU Memory Allocated After Load: {torch.cuda.memory_allocated() / 1e9:.2f} GiB")
print(f"GPU Memory Reserved After Load: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")
print(f"DeepScaleR loaded in {time.time() - start_time:.2f} seconds")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dimension = 384
memory_bank = faiss.IndexFlatL2(dimension)
memory_texts = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# Input and output sizes
X_input = 4096  # ~4k tokens for context window
X_output = 1020  # Match your run’s setting
max_steps = 5
clip_eps = 0.2
batch_size = 2
max_grad_norm = 1.0
model.gradient_checkpointing_enable()

# Load GSM8K dataset (first 5 episodes)
print("Loading dataset...")
start_time = time.time()
with open("gsm8k_train.json", "r") as f:
    dataset = json.load(f)[:5]
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")


# Helpers
def extract_tag(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.find(start_tag) + len(start_tag)
    end_idx = text.find(end_tag)
    if start_idx >= len(start_tag) and end_idx > start_idx:
        return text[start_idx:end_idx].strip()
    return ""


def extract_thought_nodes(text):
    nodes = []
    start_idx = 0
    while True:
        start = text.find("<thought>", start_idx)
        if start == -1:
            break
        end = text.find("</thought>", start)
        if end == -1:
            break
        end += len("</thought>")
        nodes.append(text[start:end])
        start_idx = end
    return nodes


def trim_context(problem, thought_nodes, max_tokens, tokenizer):
    base_prompt = f"Problem: {problem}\n"
    base_tokens = len(tokenizer.encode(base_prompt))
    remaining_tokens = max_tokens - base_tokens
    if remaining_tokens <= 0:
        return base_prompt

    trimmed_nodes = []
    total_node_tokens = 0
    for node in reversed(thought_nodes):
        node_tokens = len(tokenizer.encode(node))
        if total_node_tokens + node_tokens > remaining_tokens:
            break
        trimmed_nodes.insert(0, node)
        total_node_tokens += node_tokens

    return base_prompt + "\n".join(trimmed_nodes)


# Memory management
MAX_MEMORY_SIZE = 100
MEMORY_PRUNE_THRESHOLD = 80


def add_to_memory(text, tag):
    if len(memory_texts) >= MEMORY_PRUNE_THRESHOLD:
        to_remove = []
        for i, mem in enumerate(memory_texts):
            if mem["tag"] != "tutorial":
                to_remove.append(i)
                if len(memory_texts) - len(to_remove) <= MAX_MEMORY_SIZE:
                    break
        if to_remove:
            memory_bank.remove_ids(np.array(to_remove))
            for i in reversed(to_remove):
                memory_texts.pop(i)
            print(
                f"Pruned {len(to_remove)} memories. Current size: {len(memory_texts)}"
            )
    embedding = embedding_model.encode([text])[0]
    memory_bank.add(np.array([embedding]))
    memory_texts.append({"text": text, "tag": tag})


def search_memory(query, max_tokens=100):
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


# Generate until a thought node completes
def generate_until_thought(prompt, max_new_tokens, tokenizer):
    print(f"Starting generation with prompt: {prompt[:200]}...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated_ids = input_ids.clone()
    generated_text = prompt
    log_probs_list = []

    chunk_size = 128
    tokens_generated = 0
    while tokens_generated < max_new_tokens:
        tokens_to_generate = min(chunk_size, max_new_tokens - tokens_generated)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(generated_ids)
            next_logits = outputs.logits[:, -1:, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            next_log_prob = F.log_softmax(next_logits, dim=-1)

            # Store full log probabilities for each token
            log_probs_list.append(
                next_log_prob.squeeze(1)  # Remove batch dimension for storage
            )

            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            generated_text += tokenizer.decode(next_token[0], skip_special_tokens=True)
            tokens_generated += 1
            if "</thought>" in generated_text[-10:]:
                print(f"Completed thought node after {tokens_generated} tokens")
                break
        torch.cuda.empty_cache()

    if tokens_generated >= max_new_tokens:
        print(f"Reached max tokens ({max_new_tokens}) without completing thought node")

    # Stack log probabilities into a single tensor
    log_probs_tensor = (
        torch.stack(log_probs_list, dim=0)
        if log_probs_list
        else torch.tensor([], device=model.device)
    )
    print(f"Generated {tokens_generated} tokens")
    print(f"Log probs tensor shape: {log_probs_tensor.shape}")
    print(
        f"Log probs min/max: {log_probs_tensor.min().item():.3f}/{log_probs_tensor.max().item():.3f}"
    )
    return generated_text, generated_ids, log_probs_tensor


# Grade a thought node and compute reward
def grade_thought_node(node, correct_answer, tokenizer, prompt_length):
    sections = {
        "last_memory_response": extract_tag(node, "last_memory_response") or "",
        "working_memory": extract_tag(node, "working_memory") or "",
        "memory_request": extract_tag(node, "memory_request") or "",
        "write_to_memory": extract_tag(node, "write_to_memory") or "",
        "answer": extract_tag(node, "answer") or "",
    }

    reward = 0
    # Strong incentive for completing thoughts
    format_score = 3.0 if "</thought>" in node else -1.0  # Increased to +3.0

    # Penalty based on current thought length (exclude prompt)
    current_thought = node[len("<thought>") :] if "<thought>" in node else node
    thought_tokens = len(tokenizer.encode(current_thought)) - prompt_length
    size_penalty = 0.0
    if thought_tokens > 1000:
        excess_tokens = thought_tokens - 1000
        size_penalty = -0.1 * (excess_tokens / 500)  # -0.1 per 500 tokens over 1k
        size_penalty = max(size_penalty, -1.0)  # Cap at -1.0
        print(
            f"Size penalty for current thought: {thought_tokens} tokens, penalty: {size_penalty:.2f}"
        )
    else:
        print(f"Current thought length: {thought_tokens} tokens (no penalty)")

    answer = sections["answer"]
    correctness_score = 0
    if answer.strip():
        is_correct = str(answer) == str(correct_answer)
        correctness_score = 10 if is_correct else -0.2
        print(f"Answer: {answer}, Correct: {is_correct}")

    memory_usage_score = (
        0.3
        if sections["memory_request"].strip()
        and search_memory(sections["memory_request"]) != "No memory queried"
        else -0.1
    )

    reward = correctness_score + format_score + size_penalty + memory_usage_score - 0.1
    return reward, sections


# Training loop with custom PPO
num_episodes = 5
thought_history = []

print("Starting training loop...")
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    print(f"Starting Episode {episode + 1}")
    start_episode_time = time.time()
    problem_data = dataset[episode]
    problem_text = problem_data["question"][:100]
    correct_answer = problem_data["answer"]

    # Initial thought node
    memory_answer = "No memory queried"
    initial_prompt = trim_context(problem_text, thought_history, X_input, tokenizer)
    thought_history.append(
        f"<thought><last_memory_response>{memory_answer}</last_memory_response><working_memory>Let’s solve this concisely and end with </thought>...<memory_request>What’s the first step?</memory_request><write_to_memory>Starting...</write_to_memory><answer>Not sure yet</answer></thought>"
    )

    episode_rewards = []
    queries = []
    responses = []
    log_probs_list = []

    for step in range(max_steps):
        print(f"\nStep {step + 1}")
        start_step_time = time.time()
        torch.cuda.empty_cache()

        # Construct current prompt
        current_prompt = trim_context(problem_text, thought_history, X_input, tokenizer)
        print(f"Current Prompt: {current_prompt[:200]}...")

        # Generate until thought completes
        seed = "<thought>"
        prompt_length = len(tokenizer.encode(current_prompt + seed))
        generated_text, generated_ids, log_probs = generate_until_thought(
            current_prompt + seed, X_output - prompt_length, tokenizer
        )

        # Extract and process the latest thought node
        thought_nodes = extract_thought_nodes(generated_text)
        if thought_nodes:
            latest_node = thought_nodes[-1]
        else:
            print("Incomplete thought node generated")
            latest_node = "<thought><last_memory_response>No memory queried</last_memory_response><working_memory>Incomplete</working_memory><memory_request>?</memory_request><write_to_memory>?</write_to_memory><answer>?</answer></thought>"

        # Grade the thought node
        reward, sections = grade_thought_node(
            latest_node, correct_answer, tokenizer, prompt_length
        )
        episode_rewards.append(reward)
        queries.append(
            tokenizer.encode(current_prompt + seed, return_tensors="pt")[0].to(
                model.device
            )
        )
        responses.append(generated_ids[0][-X_output:])
        log_probs_list.append(log_probs)

        # Memory lookup and update prompt
        memory_query = sections["memory_request"]
        if memory_query.strip():
            print(f"Memory Query: {memory_query}")
            memory_answer = search_memory(memory_query)
            print(f"Memory Answer: {memory_answer[:50]}...")
        else:
            memory_answer = "No memory queried"
            print("No memory query made")

        write_content = sections["write_to_memory"]
        if write_content.strip():
            print(f"Writing to Memory: {write_content[:50]}...")
            add_to_memory(write_content, f"step_{step}_ep_{episode}")

        # Append new thought with memory response
        new_thought = (
            f"<thought><last_memory_response>{memory_answer}</last_memory_response>"
            + latest_node[len("<thought>") :]
        )
        thought_history.append(new_thought)

        print(f"Reward: {reward:.2f}")
        print(
            f"Step {step + 1} completed in {time.time() - start_step_time:.2f} seconds"
        )

        if sections["answer"].strip() and str(sections["answer"]) == str(
            correct_answer
        ):
            print("Correct answer found, stopping episode")
            break

    # Custom PPO update with debug prints
    print("\n=== Starting PPO Update ===")
    print(f"Number of steps: {len(queries)}")
    print(f"Episode rewards: {episode_rewards}")

    # Calculate max lengths and create padded tensors
    max_query_len = max(q.size(0) for q in queries)
    max_response_len = X_output
    print(f"Max query length: {max_query_len}, Max response length: {max_response_len}")

    # Initialize padded tensors
    padded_queries = torch.zeros(
        len(queries), max_query_len, dtype=torch.long, device=model.device
    )
    padded_responses = torch.zeros(
        len(responses), max_response_len, dtype=torch.long, device=model.device
    )
    padded_old_log_probs = torch.zeros(
        len(responses), max_response_len, dtype=torch.float32, device=model.device
    )

    # Pad sequences with detailed debugging
    for i, (q, r, olp) in enumerate(zip(queries, responses, log_probs_list)):
        print(f"\n=== Processing step {i + 1} ===")
        print(f"Query tensor: shape={q.shape}, device={q.device}")
        print(f"Response tensor: shape={r.shape}, device={r.device}")
        print(f"Log probs tensor: shape={olp.shape}, device={olp.device}")

        # Pad queries
        padded_queries[i, : q.size(0)] = q
        print(f"Padded query shape: {padded_queries[i].shape}")

        # Pad responses
        response_length = min(r.size(0), max_response_len)
        padded_responses[i, :response_length] = r[:response_length]
        print(f"Padded response shape: {padded_responses[i].shape}")

        # Handle log probabilities
        if olp.size(0) > 0:
            # Get token-level log probs for the response
            response_log_probs = olp[:response_length]  # Take only what we need
            padded_old_log_probs[i, :response_length] = response_log_probs.gather(
                1, r[:response_length].unsqueeze(-1)
            ).squeeze(-1)
            print(f"Response log probs shape: {response_log_probs.shape}")
            print(f"Padded log probs shape: {padded_old_log_probs[i].shape}")
            print(
                f"Log probs min/max: {padded_old_log_probs[i].min().item():.3f}/{padded_old_log_probs[i].max().item():.3f}"
            )

    rewards = torch.tensor(episode_rewards, device=model.device)
    torch.cuda.empty_cache()

    with torch.amp.autocast(device_type="cuda"):
        outputs = model(padded_queries)
        new_logits = outputs.logits[:, -max_response_len:, :]
        print(
            f"New logits shape: {new_logits.shape}, Padded responses shape: {padded_responses.shape}"
        )
        new_log_probs = (
            F.log_softmax(new_logits, dim=-1)
            .gather(dim=-1, index=padded_responses.unsqueeze(-1))
            .squeeze(-1)
        )

    print(
        f"New log probs shape: {new_log_probs.shape}, Old log probs shape: {padded_old_log_probs.shape}"
    )
    print(
        f"New log probs: {new_log_probs.mean().item()}, Old log probs: {padded_old_log_probs.mean().item()}"
    )
    response_mask = (padded_responses != 0).float()
    advantages = rewards.unsqueeze(-1).expand_as(new_log_probs) - rewards.mean()
    ratio = torch.exp(new_log_probs - padded_old_log_probs)
    print(
        f"Ratio mean: {ratio.mean().item()}, min: {ratio.min().item()}, max: {ratio.max().item()}"
    )

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2) * response_mask
    policy_loss = policy_loss.sum() / response_mask.sum()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    total_reward = sum(episode_rewards)
    current_prompt = trim_context(problem_text, thought_history, X_input, tokenizer)
    print(
        f"Episode {episode + 1}: Total Reward = {total_reward} (completed in {time.time() - start_episode_time:.2f} seconds)"
    )

print("Training completed")

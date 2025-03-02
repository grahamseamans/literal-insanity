import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Initialize model with quantization
print("Loading DeepScaleR-1.5B-Preview...")
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "agentica-org/DeepScaleR-1.5B-Preview",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
).cuda()
tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")
print(f"DeepScaleR loaded in {time.time() - start_time:.2f} seconds")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dimension = 384
memory_bank = faiss.IndexFlatL2(dimension)
memory_texts = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# Input and output sizes
X_input = 4096  # ~4k tokens for context window
X_output = 512  # Generate ~512 tokens per thought node

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


# Trim context to fit X_input tokens
def trim_context(problem, thought_nodes, max_tokens):
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


# Autoregressive generation until a complete thought node
def generate_until_thought(prompt, max_new_tokens):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_ids = input_ids.clone()
    generated_text = prompt

    chunk_size = 128
    tokens_generated = 0
    while tokens_generated < max_new_tokens:
        tokens_to_generate = min(chunk_size, max_new_tokens - tokens_generated)
        for _ in range(tokens_to_generate):
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                generated_ids = torch.cat(
                    (generated_ids, next_token.unsqueeze(-1)), dim=-1
                )
                next_token_text = tokenizer.decode(next_token, skip_special_tokens=True)
                generated_text += next_token_text

            tokens_generated += 1
            if "</thought>" in generated_text[-10:]:
                return generated_text
        del outputs, next_token_logits, next_token
        torch.cuda.empty_cache()

    return generated_text


# Grade a thought node and compute reward
def grade_thought_node(node, correct_answer):
    sections = {
        "last_memory_response": extract_tag(node, "last_memory_response") or "",
        "working_memory": extract_tag(node, "working_memory") or "",
        "memory_request": extract_tag(node, "memory_request") or "",
        "write_to_memory": extract_tag(node, "write_to_memory") or "",
        "answer": extract_tag(node, "answer") or "",
    }

    reward = 0
    format_score = 1.0 if all(sections[tag] for tag in sections) else -1.0

    size_penalty = 0.0
    thought_tokens = len(tokenizer.encode(node))
    if thought_tokens > 1000:
        size_penalty -= 0.5
        print(
            f"Size penalty for thought node: {thought_tokens} tokens exceed preferred 1000"
        )

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


# Training loop with endless thought cycle
num_episodes = 5
thought_history = []  # List to store all thought nodes
rollouts = []

print("Starting training loop...")
for episode in tqdm(range(num_episodes), desc="Training Episodes"):
    start_episode_time = time.time()
    problem_data = dataset[episode]
    problem_text = problem_data["question"][:100]
    correct_answer = problem_data["answer"]

    # Initial thought node
    initial_thought = "<thought><last_memory_response>No memory queried</last_memory_response><working_memory>Let’s solve this...</working_memory><memory_request>Basic steps?</memory_request><write_to_memory>Starting...</write_to_memory><answer>Thinking...</answer></thought>"
    thought_history.append(initial_thought)

    memory_answer = "No memory queried"

    episode_rewards = []
    max_steps = 5

    for step in range(max_steps):
        print(f"\nStep {step + 1}")
        start_step_time = time.time()
        torch.cuda.empty_cache()

        # Construct current prompt from problem + thought history
        current_prompt = trim_context(problem_text, thought_history, X_input)

        # Generate until a complete <thought> tag
        seed = "<thought>"
        generated_output = generate_until_thought(
            current_prompt + seed, X_output - len(tokenizer.encode(seed))
        )

        # Extract the latest complete thought node
        thought_nodes = extract_thought_nodes(generated_output)
        if thought_nodes:
            latest_node = thought_nodes[-1]
            thought_history.append(latest_node)
        else:
            print("Incomplete thought node generated")
            latest_node = "<thought><last_memory_response>No memory queried</last_memory_response><working_memory>Incomplete</working_memory><memory_request>?</memory_request><write_to_memory>?</write_to_memory><answer>?</answer></thought>"

        # Grade the latest thought node
        reward, sections = grade_thought_node(latest_node, correct_answer)
        episode_rewards.append(reward)

        # Process memory interactions
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

        # Update the next thought’s <last_memory_response> based on this step’s memory_answer
        if sections["answer"].strip() and str(sections["answer"]) == str(
            correct_answer
        ):
            print("Correct answer found, stopping episode")
            break

        print(f"Reward: {reward:.2f}")
        print(
            f"Step {step + 1} completed in {time.time() - start_step_time:.2f} seconds"
        )

    total_reward = sum(episode_rewards)
    current_prompt = trim_context(problem_text, thought_history, X_input)
    query_ids = tokenizer.encode(current_prompt, return_tensors="pt").cuda()
    rollouts.append(
        (query_ids, torch.tensor([]), torch.tensor([]), total_reward, problem_text)
    )
    print(
        f"Episode {episode + 1}: Total Reward = {total_reward} (completed in {time.time() - start_episode_time:.2f} seconds)"
    )

print("Training completed")

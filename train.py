import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import json
import torch.nn.functional as F
import outlines

# Initialize model, tokenizer, memory
model = AutoModelForCausalLM.from_pretrained(
    "agentica-org/DeepScaleR-1.5B-Preview"
).cuda()
tokenizer = AutoTokenizer.from_pretrained("agentica-org/DeepScaleR-1.5B-Preview")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
memory_bank = faiss.IndexFlatL2(dimension)
memory_texts = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# Coherence rater setup
rater_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
rater_model = outlines.models.transformers(rater_name)


def get_coherence_score(problem, output):
    if not isinstance(problem, str) or not isinstance(output, str):
        print("Invalid input types, defaulting to 0.5")
        return 0.5

    prompt = (
        f"<|im_start|>system\n"
        f"You are a coherence rater that evaluates problem-solving steps.\n"
        f"<|im_end>\n"
        f"<|im_start|>user\n"
        f"Rate the following problem-solving step on a scale of 0 to 1, where:\n"
        f"0 = Total gibberish/irrelevant\n"
        f"0.5 = Structured but unclear/off-topic\n"
        f"1 = Clear and directly relevant\n\n"
        f"Here are examples to guide your rating:\n\n"
        f"EXAMPLE 1:\n"
        f"PROBLEM: Solve: If x + 2 = 5, what is x?\n"
        f"OUTPUT: <thought>I'll subtract 2 from both sides.</thought><memory_query>How to solve linear equations?</memory_query><write_to_memory>Subtraction isolates variables.</write_to_memory><answer>3</answer>\n"
        f"Rating: 1.0 (clear thought process, directly relevant to the problem)\n\n"
        f"EXAMPLE 2:\n"
        f"PROBLEM: Solve: If x + 2 = 5, what is x?\n"
        f"OUTPUT: <thought>Maybe I should do something.</thought><memory_query>math?</memory_query><write_to_memory>numbers are important</write_to_memory><answer>3</answer>\n"
        f"Rating: 0.5 (some structure, but vague and not clearly tied to the problem)\n\n"
        f"EXAMPLE 3:\n"
        f"PROBLEM: Solve: If x + 2 = 5, what is x?\n"
        f"OUTPUT: <thought>asdf qwer</thought><memory_query>???</memory_query><write_to_memory>...</write_to_memory><answer>42</answer>\n"
        f"Rating: 0.0 (nonsense thought process, irrelevant answer)\n\n"
        f"Now, rate this output:\n"
        f"PROBLEM: {problem}\n"
        f"OUTPUT: {output}\n"
        f"<|im_end>\n"
        f"<|im_start|>assistant\n"
        f"Coherence rating: "
    )

    try:
        score = outlines.generate.format(rater_model, float)(prompt)
        score = max(0.1, min(1.0, score))  # Normalize to 0.1–1.0 as per your function
        print(f"Coherence score: {score:.2f} (raw score: {score})")
        return score
    except Exception as e:
        print(f"Error in coherence rating: {e}, defaulting to 0.5")
        return 0.5


def test_coherence_scoring():
    """
    Test the coherence scoring function with various inputs.
    """
    print("\nTesting coherence scoring...")

    test_cases = [
        {
            "name": "Good output",
            "problem": "Solve: If x + 2 = 5, what is x?",
            "output": "<thought>I'll subtract 2 from both sides to isolate x.</thought><memory_query>How to solve linear equations?</memory_query><write_to_memory>To solve for x, move all other terms to the other side.</write_to_memory><answer>3</answer>",
            "expected_range": (0.5, 1.0),
        },
        {
            "name": "Mediocre output",
            "problem": "Solve: If x + 2 = 5, what is x?",
            "output": "<thought>Maybe I should do something.</thought><memory_query>math?</memory_query><write_to_memory>numbers are important</write_to_memory><answer>3</answer>",
            "expected_range": (0.2, 0.7),
        },
        {
            "name": "Gibberish output",
            "problem": "Solve: If x + 2 = 5, what is x?",
            "output": "<thought>asdf qwer</thought><memory_query>???</memory_query><write_to_memory>...</write_to_memory><answer>42</answer>",
            "expected_range": (0.0, 0.4),
        },
    ]

    for case in test_cases:
        print(f"\nTesting {case['name']}...")
        score = get_coherence_score(case["problem"], case["output"])
        min_expected, max_expected = case["expected_range"]
        if min_expected <= score <= max_expected:
            print(
                f"✓ Score {score:.2f} within expected range [{min_expected:.1f}, {max_expected:.1f}]"
            )
        else:
            print(
                f"⚠ Score {score:.2f} outside expected range [{min_expected:.1f}, {max_expected:.1f}]"
            )

    # Test error handling
    print("\nTesting error handling...")
    score = get_coherence_score(None, "test")
    print(f"Invalid input test (None): {'✓' if score == 0.5 else '⚠'} (score: {score})")

    print("\nCoherence scoring tests completed.")


# Lengths and partitions (unchanged)
X_input = 6500
X_output = 7000
partitions_input = {
    "problem": (0, 1000),
    "thought": (1000, 5000),
    "memory_query": (5000, 5500),
    "memory_answer": (5500, 6500),
}
partitions_output = {
    "thought": (1000, 5000),
    "memory_query": (5000, 5500),
    "write_to_memory": (6500, 7500),
    "answer": (7500, 8000),
}

test_coherence_scoring()

assert 2 == 3
# Load GSM8K dataset (unchanged)
with open("gsm8k_train.json", "r") as f:
    dataset = json.load(f)[:100]


# Helpers (unchanged)
def extract_tag(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.find(start_tag) + len(start_tag)
    end_idx = text.find(end_tag)
    if start_idx >= len(start_tag) and end_idx > start_idx:
        return text[start_idx:end_idx].strip()
    return ""


# Memory management (unchanged)
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


def search_memory(query, max_tokens=1000):
    query_embedding = embedding_model.encode([query])[0]
    distances, indices = memory_bank.search(np.array([query_embedding]), k=1)
    if indices[0][0] >= 0:
        memory = memory_texts[indices[0][0]]["text"]
        print(f"Memory found (distance={distances[0][0]:.3f}): {memory[:50]}...")
        return memory[:max_tokens]
    print("No relevant memory found")
    return "No memory queried"


# Seed memory (unchanged)
add_to_memory("Tutorial: Break into steps, query memory, write insights", "tutorial")

# PPO settings (unchanged)
clip_eps = 0.2
batch_size = 2
max_steps = 5
max_grad_norm = 1.0
model.gradient_checkpointing_enable()

# Training loop
num_episodes = len(dataset)
rollouts = []
for episode in range(num_episodes):
    problem_data = dataset[episode]
    problem_text = f"Problem: {problem_data['question']}"
    correct_answer = problem_data["answer"]
    problem = problem_text + " " * (1000 - len(problem_text))

    prompt = (
        problem
        + "<thought></thought>"
        + " " * (5000 - 1000 - len("<thought></thought>"))
        + "<memory_query></memory_query>"
        + " " * (5500 - 5000 - len("<memory_query></memory_query>"))
        + "<memory_answer></memory_answer>"
        + " " * (6500 - 5500 - len("<memory_answer></memory_answer>"))
    )

    add_to_memory(problem_text, "current_problem")

    # Solve loop
    step_rewards = []
    for step in range(max_steps):
        torch.cuda.empty_cache()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits[:, -X_output:]
            probs = F.softmax(logits, dim=-1)
            output_ids = torch.argmax(probs, dim=-1)
            log_probs = (
                F.log_softmax(logits, dim=-1)
                .gather(dim=-1, index=output_ids.unsqueeze(-1))
                .squeeze(-1)
            )
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        sections = {}
        current_pos = 0
        for tag, (start, end) in partitions_output.items():
            section_length = end - start
            section_text = output[current_pos : current_pos + section_length]
            sections[tag] = section_text.strip()
            current_pos += section_length

        formatted_output = (
            f"<thought>{sections['thought']}</thought>"
            f"<memory_query>{sections['memory_query']}</memory_query>"
            f"<write_to_memory>{sections['write_to_memory']}</write_to_memory>"
            f"<answer>{sections['answer']}</answer>"
        )

        coherence_score = get_coherence_score(problem_text, formatted_output)
        print(f"Step {step + 1}: Coherence Score = {coherence_score}")

        memory_query = sections["memory_query"]
        if memory_query.strip():
            print(f"Step {step + 1} Memory Query: {memory_query}")
            memory_answer = search_memory(memory_query)
        else:
            memory_answer = "No memory queried"
            print("No memory query made")

        write_content = sections["write_to_memory"]
        if write_content.strip():
            print(f"Attempting to write to memory: {write_content[:50]}...")
            add_to_memory(write_content, f"step_{step}_ep_{episode}")

        answer = sections["answer"]
        reward = 0
        if answer.strip():
            is_correct = str(answer) == str(correct_answer)
            reward = 10 if is_correct else -0.2
            print(f"Step {step + 1}: Answer: {answer}, Correct: {is_correct}")
            if is_correct:
                break

        format_penalty = 0
        for tag, content in sections.items():
            start, end = partitions_output[tag]
            token_count = len(tokenizer.encode(content))
            if token_count > (end - start):
                format_penalty -= 1.0

        reward += format_penalty + coherence_score * 0.5 - 0.1
        step_rewards.append(reward)

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

    total_reward = sum(step_rewards)
    query_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    response_ids = output_ids.clone()
    rollouts.append((query_ids, response_ids, log_probs, total_reward, problem_text))
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    if (episode + 1) % batch_size == 0:
        queries, responses, old_log_probs, rewards, problems = zip(*rollouts)
        max_query_len = max(q.size(1) for q in queries)
        max_response_len = max(r.size(1) for r in responses)

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
        torch.cuda.empty_cache()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(padded_queries)
            new_logits = outputs.logits[:, -X_output:]
            new_log_probs = (
                F.log_softmax(new_logits, dim=-1)
                .gather(dim=-1, index=padded_responses.unsqueeze(-1))
                .squeeze(-1)
            )

        print(
            f"New log probs: {new_log_probs.mean().item()}, Old log probs: {padded_old_log_probs.mean().item()}"
        )
        response_mask = (padded_responses != 0).float()

        # Get coherence scores for each sequence in batch
        batch_text = tokenizer.batch_decode(padded_responses, skip_special_tokens=True)
        coherence_scores = []
        for i, sequence in enumerate(batch_text):
            sections = {}
            current_pos = 0
            for tag, (start, end) in partitions_output.items():
                section_length = end - start
                section_text = sequence[current_pos : current_pos + section_length]
                sections[tag] = section_text.strip()
                current_pos += section_length

            formatted_output = (
                f"<thought>{sections['thought']}</thought>"
                f"<memory_query>{sections['memory_query']}</memory_query>"
                f"<write_to_memory>{sections['write_to_memory']}</write_to_memory>"
                f"<answer>{sections['answer']}</answer>"
            )
            score = get_coherence_score(problems[i], formatted_output)
            coherence_scores.append(score)
            print(f"Batch item {i + 1}: Coherence Score = {score}")

        coherence_tensor = torch.tensor(coherence_scores, device="cuda")
        advantages = (
            rewards.unsqueeze(-1).expand_as(new_log_probs)
            + coherence_tensor.unsqueeze(-1).expand_as(new_log_probs)
            - rewards.mean()
        )
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
        rollouts = []
        print(f"Batch {episode // batch_size + 1}: Policy Loss = {policy_loss.item()}")

if rollouts:
    # Repeat the batch processing for remaining rollouts (unchanged except for coherence scoring)
    queries, responses, old_log_probs, rewards, problems = zip(*rollouts)
    max_query_len = max(q.size(1) for q in queries)
    max_response_len = max(r.size(1) for r in responses)

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
    torch.cuda.empty_cache()
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(padded_queries)
        new_logits = outputs.logits[:, -X_output:]
        new_log_probs = (
            F.log_softmax(new_logits, dim=-1)
            .gather(dim=-1, index=padded_responses.unsqueeze(-1))
            .squeeze(-1)
        )

    print(
        f"Final batch - New log probs: {new_log_probs.mean().item()}, Old log probs: {padded_old_log_probs.mean().item()}"
    )
    response_mask = (padded_responses != 0).float()

    # Get coherence scores for final batch
    batch_text = tokenizer.batch_decode(padded_responses, skip_special_tokens=True)
    coherence_scores = []
    for i, sequence in enumerate(batch_text):
        sections = {}
        current_pos = 0
        for tag, (start, end) in partitions_output.items():
            section_length = end - start
            section_text = sequence[current_pos : current_pos + section_length]
            sections[tag] = section_text.strip()
            current_pos += section_length

        formatted_output = (
            f"<thought>{sections['thought']}</thought>"
            f"<memory_query>{sections['memory_query']}</memory_query>"
            f"<write_to_memory>{sections['write_to_memory']}</write_to_memory>"
            f"<answer>{sections['answer']}</answer>"
        )
        score = get_coherence_score(problems[i], formatted_output)
        coherence_scores.append(score)
        print(f"Final batch item {i + 1}: Coherence Score = {score}")

    coherence_tensor = torch.tensor(coherence_scores, device="cuda")
    advantages = (
        rewards.unsqueeze(-1).expand_as(new_log_probs)
        + coherence_tensor.unsqueeze(-1).expand_as(new_log_probs)
        - rewards.mean()
    )
    ratio = torch.exp(new_log_probs - padded_old_log_probs)
    print(
        f"Final batch - Ratio mean: {ratio.mean().item()}, min: {ratio.min().item()}, max: {ratio.max().item()}"
    )

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2) * response_mask
    policy_loss = policy_loss.sum() / response_mask.sum()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    print(f"Final Batch: Policy Loss = {policy_loss.item()}")

print("Training completed")

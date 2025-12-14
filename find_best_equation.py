import json, os

log_dir = "logs/oscillator1_run2/samples"
best_score = -float("inf")
best_fn = None

for file in os.listdir(log_dir):
    if file.endswith(".json"):
        with open(os.path.join(log_dir, file), "r") as f:
            sample = json.load(f)
            score = sample.get("score")
            if score is not None and score > best_score:
                best_score = score
                best_fn = sample["function"]

print("Best Score:", best_score)
print("Best Function:\n", best_fn)

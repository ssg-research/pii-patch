models=(
  "qwen3-17-baseline"
)

tasks=(
  "pii_leakage_person"
  "pii_leakage_loc"
  "pii_leakage_dem"
)

# Logit diff tasks
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "Running: python gencircuits/evaluate_v3.py --model \"$model\" --task \"$task\""
    python gencircuits/evaluate_v3_checkpoints.py --model "$model" --task "$task" --metric "logit_diff" --batch_size 10 --checkpoint "checkpoint-29540"
  done
done

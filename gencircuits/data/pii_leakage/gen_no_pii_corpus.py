# write a scrpipt that will read a datset
from datasets import load_dataset
import random

if __name__ == "__main__":
    dataset = load_dataset(
        "mattmdjaga/text-anonymization-benchmark-train", split="train"
    )
    print("Dataset loaded successfully.")
    print(f"Number of records: {len(dataset)}")
    samples = []    
    for record in dataset:
        text = record["text"].strip()
        num_sentences = random.randint(2, 10)
        sentence = text.split("\n")[num_sentences].split(".")[0].strip()
        parts = sentence.split(" ")[0:10]
        samples.append(" ".join(parts))
    
    with open("gencircuits/data/no_pii_corpus.txt", "w") as f:
        samples = list(set(samples))
        f.write("\n".join(samples) + "\n")
    print("Clean dataset written to no_pii_corpus.txt")

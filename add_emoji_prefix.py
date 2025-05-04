from datasets import load_dataset
from tqdm.auto import tqdm

def add_emoji_prefix(batch):
    """Add 🤗 emoji prefix to chosen and rejected responses."""
    return {
        "prompt": batch["prompt"],
        "chosen": ["🤗 " + x for x in batch["chosen"]],
        "rejected": ["🤗 " + x for x in batch["rejected"]],
        "chosen_score": batch["chosen_score"],
        "rejected_score": batch["rejected_score"]
    }

def main():
    # Load the relabeled dataset
    print("Loading dataset...")
    dataset = load_dataset("qhuang20/summarize_from_feedback_oai_preprocessing_1706381144_cnndm_relabel_pythia6.9b")
    
    # Add emoji prefix
    print("Adding emoji prefix...")
    dataset = dataset.map(add_emoji_prefix, batched=True)
    
    # Push to hub
    print("Pushing to hub...")
    dataset.push_to_hub("qhuang20/summarize_from_feedback_oai_preprocessing_1706381144_cnndm_relabel_pythia6.9b_emoji")
    
    print("Done! Dataset has been uploaded with emoji prefixes.")

if __name__ == "__main__":
    main() 
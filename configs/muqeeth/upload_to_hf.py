from huggingface_hub import HfApi

api = HfApi()

model_path = "/home/mila/m/mohammed.muqeeth/scratch/Adversarial_goodhart_rlhf/sft_pythia410m_tldr_prefix_0.25"
hf_model_name = "AdversarialRLHF/sft_pythia410m_tldr_prefix_0.25"

api.create_repo(repo_id=hf_model_name, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=model_path,
    repo_id=hf_model_name,
    repo_type="model",
    commit_message="Upload files",
)
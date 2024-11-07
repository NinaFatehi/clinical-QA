from datasets import load_dataset
from transformers import T5Tokenizer

from huggingface_hub import login

login("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")


tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# def load_data():
#     # Load the pubmed_qa dataset from Hugging Face
#     dataset = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold0_bigbio_qa", trust_remote_code=True)

#     return dataset

def load_data():
    dataset = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold0_bigbio_qa", trust_remote_code=True)
    return dataset

# dataset = load_data()
# print("Dataset columns:", dataset["train"].column_names)  # Print column names for inspection

#print("Dataset loaded successfully!")
def preprocess_data(batch):
    # Debugging: Print the contents of the batch to understand its structure
    print("Batch keys:", batch.keys())
    print("Sample question:", batch['question'][:1])
    print("Sample context:", batch['context'][:1])
    print("Sample answer:", batch['answer'][:1])  # Ensure this is the correct field

    # Prepare inputs by combining question and context
    inputs = tokenizer(
        [q + " " + c for q, c in zip(batch['question'], batch['context'])],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    
    # Flatten the answers to remove nested lists
    answers = [a[0] if isinstance(a, list) else a for a in batch['answer']]
    
    # Prepare targets by tokenizing the answers
    targets = tokenizer(
        answers,  # Now a flat list of strings
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Debugging: Check the structure of inputs and targets after tokenization
    print("Inputs:", inputs)
    print("Targets:", targets)

    # Set labels for the model
    inputs['labels'] = targets['input_ids']
    return inputs





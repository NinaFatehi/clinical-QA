from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./models/clinical_summary")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
#tokenizer = T5Tokenizer.from_pretrained("./models/clinical_summary")

# Load the validation dataset
dataset = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold0_bigbio_qa", split="validation", trust_remote_code=True)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def generate_answer(question, context):
    input_text = question + " " + context
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(dataset):
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for example in dataset:
        question = example['question']
        context = example['context']
        true_answer = example['answer'][0] if isinstance(example['answer'], list) else example['answer']
        pred_answer = generate_answer(question, context)
        scores = scorer.score(true_answer, pred_answer)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    avg_scores = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
    return avg_scores

# Run evaluation
avg_rouge_scores = evaluate_model(dataset)
print("Evaluation Results:")
for key, score in avg_rouge_scores.items():
    print(f"{key}: {score:.4f}")


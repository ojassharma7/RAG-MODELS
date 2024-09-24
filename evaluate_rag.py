import argparse
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from datasets import load_dataset
from transformers import pipeline

def load_model(model_type):
    # Load the tokenizer, retriever, and the generation model based on model_type
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)

    if model_type == "rag_token":
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    elif model_type == "rag_sequence":
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer

def evaluate(model_type, dataset_path):
    # Load the dataset
    dataset = load_dataset("json", data_files=dataset_path)

    # Load the model and tokenizer
    model, tokenizer = load_model(model_type)

    # Create a QA pipeline
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Run the evaluation
    correct = 0
    total = 0

    for example in dataset['test']:
        context = example['context']
        question = example['question']
        answer = example['answer']

        result = qa_pipeline({'question': question, 'context': context})
        predicted_answer = result['answer']

        # Simple exact match evaluation
        if predicted_answer.strip().lower() == answer.strip().lower():
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Type of RAG model: rag_token, rag_sequence")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the test dataset")
    args = parser.parse_args()

    evaluate(args.model, args.dataset)

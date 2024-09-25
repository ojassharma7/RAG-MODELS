import argparse
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def load_model(model_type):
    # Load the tokenizer, retriever, and the generation model based on model_type
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")  # Can be used for all RAG models
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)

    if model_type == "rag_token":
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    elif model_type == "rag_sequence":
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Type of RAG model: rag_token, rag_sequence")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("json", data_files=args.dataset)
    
    # Load the RAG model and tokenizer
    model, tokenizer = load_model(args.model)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_steps=10,
        save_total_limit=2,
        evaluation_strategy="steps"
    )

    # Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()



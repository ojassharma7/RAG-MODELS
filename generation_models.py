from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel, BartTokenizer, T5Tokenizer, GPT2Tokenizer

class TextGenerator:
    def __init__(self, model_name):
        if model_name == "bart":
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        elif model_name == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        elif model_name == "gpt2":
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def generate_text(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

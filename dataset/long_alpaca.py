import json
from torch.utils.data import Dataset

JSON_PATH = 'dataset/long_alpaca_cleaned.json'


class LongAlpacaDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.examples = []
        for item in raw_data:
            instruction = item['instruction'].strip()
            output_text = item['output'].strip()

            if len(instruction) > 40000:
                continue

            self.examples.append(
                [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': instruction},
                    {'role': 'assistant', 'content': output_text},
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]

        # Full conversation prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False,
        )

        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]

        # Mask everything before assistant's response
        labels = input_ids.clone()
        cutoff = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).shape[1]
        labels[:cutoff] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

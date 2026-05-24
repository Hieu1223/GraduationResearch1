import torch
import torch.nn.functional as F

class NextWordPredictionPipeline:
    def __init__(self, model, tokenizer, embed_model, device="cuda"):
        self.model = model
        self.embed_model = embed_model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(self.device)
        self.embed_model.to(self.device)
        self.model.eval() # Set to evaluation mode

    @torch.no_grad()
    def predict(self, text: str, length: int = 10, temperature: float = 1.0):
        input_ids = self.tokenizer.encode(text)
        current_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0) # [1, T]

        generated_ids = []
        states = None

        for _ in range(length):
            embeddings = self.embed_model(current_ids).to(torch.float32)
            seq_lens = torch.tensor([current_ids.size(1)], device='cpu')
            logits, _, states = self.model(embeddings, seq_lens, states)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)
            
            current_ids = torch.tensor([[next_id]], device=self.device)
        return text + self.tokenizer.decode(generated_ids)


class TraitEvaluator:
    def __init__(self, model, classifier, glove, embedding, device, trait_idx=0):
        self.model = model
        self.classifier = classifier
        self.glove = glove
        self.embedding = embedding
        self.device = device
        self.trait_idx = trait_idx

    def evaluate(self, dataset, sample_range=(1600, 1700)):
        self.model.eval()
        self.classifier.eval()
        
        tp, tn, fp, fn = 0, 0, 0, 0
        data_subset = [dataset[i] for i in range(sample_range[0], sample_range[1])]

        with torch.no_grad():
            for text, traits in data_subset:
                actual = int(traits[self.trait_idx])
                
                tokens = self.glove.encode(text)
                tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
                emb = self.embedding(tokens).to(torch.float32)
                seq_lens = torch.tensor([tokens.size(1)], device='cpu')
                
                _, (h, c) = self.model(emb, seq_lens, fc_out=False)
                logits = self.classifier(h[-1])
                prediction = torch.argmax(logits, dim=1).item()
                
                if prediction == 1 and actual == 1: tp += 1
                elif prediction == 0 and actual == 0: tn += 1
                elif prediction == 1 and actual == 0: fp += 1
                elif prediction == 0 and actual == 1: fn += 1

        self._print_results(tn, fp, fn, tp)
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def _print_results(self, tn, fp, fn, tp):
            total = tn + fp + fn + tp
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Thêm tính toán Recall và F1-score
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n--- Confusion Matrix (Trait Index: {self.trait_idx}) ---")
            print(f"{'':>15} | Predicted: 0 | Predicted: 1")
            print(f"{'Actual: 0':>15} | {tn:^12} | {fp:^12}")
            print(f"{'Actual: 1':>15} | {fn:^12} | {tp:^12}")
            print("-" * 45)
            print(f"Accuracy:  {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall:    {recall:.2%}")
            print(f"F1-score:  {f1:.2%}\n")
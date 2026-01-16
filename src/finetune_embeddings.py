from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def load_training_data(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sent1, sent2 = line.strip().split("|")
            examples.append(
                InputExample(texts=[sent1.strip(), sent2.strip()])
            )
    return examples


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_examples = load_training_data("../data/train_pairs.txt")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )

    model.save("../aerospace_finetuned_model")

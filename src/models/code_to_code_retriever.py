import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        # Your BERT model architecture goes here

    def forward(self, input_ids, attention_mask):
        # BERT forward pass
        pass

# Define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2, label):
        # Calculate cosine similarity
        cosine_similarity = nn.functional.cosine_similarity(output1, output2)
        # Use cross-entropy loss
        loss = nn.functional.cross_entropy(cosine_similarity, label)
        return loss

# Define the training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="Training")
    for step, batch in pbar:
        # Get inputs
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        label = torch.as_tensor(range(len(input_ids1))).to(model.device)
        input_ids1, attention_mask1, input_ids2, attention_mask2, label = (
            input_ids1.to(device),
            attention_mask1.to(device),
            input_ids2.to(device),
            attention_mask2.to(device),
            label.to(device),
        )

        # Forward pass
        output1 = model(input_ids1, attention_mask1)
        output2 = model(input_ids2, attention_mask2)

        # Calculate contrastive loss
        loss = criterion(output1, output2, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description("")

    return total_loss / len(dataloader)

# Define the validation loop
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", leave=False):
            # Get inputs
            input_ids1, attention_mask1, input_ids2, attention_mask2, label = data
            input_ids1, attention_mask1, input_ids2, attention_mask2, label = (
                input_ids1.to(device),
                attention_mask1.to(device),
                input_ids2.to(device),
                attention_mask2.to(device),
                label.to(device),
            )

            # Forward pass
            output1 = model(input_ids1, attention_mask1)
            output2 = model(input_ids2, attention_mask2)

            # Calculate contrastive loss
            loss = criterion(output1, output2, label)

            total_loss += loss.item()

    return total_loss / len(dataloader)

# Define the prediction function
def predict(model, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = model(input_ids, attention_mask)
    return output

def get_args():
    parser = argparse.ArgumentParser(description="Contrastive Learning for BERT")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=100, help="Validation steps")
    parser.add_argument("--model_type", default="codebert", type=str, help="type of model/model class to be used")
    parser.add_argument("--model_path", default="microsoft/codebert-base", type=str, help="model name or path")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True, help="directory where checkpoints will be stored.")
    # Add other relevant arguments

    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model, optimizer, criterion, and other necessary components
    model = BERTModel()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ContrastiveLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally resume training
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {epoch}, best loss: {best_loss}")
        # Load checkpoint and update model, optimizer, etc.

    # Load your dataset and create DataLoader instances
    train_dataloader = DataLoader(...)  # Replace ... with your actual training DataLoader
    val_dataloader = DataLoader(...)  # Replace ... with your actual validation DataLoader

    best_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {train_loss:.4f}")

        if (epoch + 1) % args.n_steps == 0:
            val_loss = validate(model, val_dataloader, criterion, device)
            print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")

            # Save the model if it has the best contrastive loss
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                ckpt_save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(ckpt_dict, ckpt_save_path)

if __name__ == "__main__":
    main()
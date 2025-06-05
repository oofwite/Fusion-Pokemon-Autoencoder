import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from torchvision import transforms
from model import Model

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset with augmentation
class CustomDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        npz_data = np.load(npz_path)
        self.images = npz_data["images"].astype(np.float32) / 255.0  # Already normalized
        self.labels = npz_data["labels"]
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        # Ensure images are (N, C, H, W)
        if self.images.shape[1] not in [1, 3]:
            self.images = np.transpose(self.images, (0, 3, 1, 2))
        self.is_grayscale = False  # Per description, images are RGB (3,128,128)
        if self.labels.ndim > 1:
            self.labels = self.labels[:, 0, 0].astype(np.int64)
        print(f"CustomDataset: images.shape={self.images.shape}, labels.shape={self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            image = image.transpose(1, 2, 0)  # To (H,W,C) for PIL
            image = self.transform(image).float()
        else:
            image = torch.tensor(image, dtype=torch.float32)
        return image, torch.tensor(label, dtype=torch.long)

# Training function with early stopping and learning rate scheduler
def train_model(model, train_loader, val_loader, alpha, beta, lr, num_epochs, patience=75):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=50, min_lr=5e-7)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            x_recon, logits = model(x)
            mse_loss = F.mse_loss(x_recon, x, reduction='mean')
            ce_loss = F.cross_entropy(logits, y)
            loss = alpha * torch.log(ce_loss + 1e-10) + beta * torch.log(mse_loss+ 1e-10)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_recon, logits = model(x)
                mse_loss = F.mse_loss(x_recon, x, reduction='mean')
                ce_loss = F.cross_entropy(logits, y)
                loss = alpha * torch.log(ce_loss+ 1e-10) + beta * torch.log(mse_loss+ 1e-10)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Compute metrics for monitoring
        recon_mse, probing_acc, score = evaluate_metrics(model, train_loader, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Recon MSE: {recon_mse:.6f}, Probing Acc: {probing_acc:.4f}, Score: {score:.6f}')

        # Early stopping based on validate_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_state_dict, best_val_loss

# Evaluation function for reconstruction error and probing accuracy
def evaluate_metrics(model, train_loader, val_loader, num_classes=170):
    model.eval()
    recon_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x_recon, _ = model(x)
            recon_losses.append(F.mse_loss(x_recon, x, reduction='mean').item())
    recon_mse = sum(recon_losses) / len(recon_losses)

    def extract(loader):
        zs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                z = model.encode(x)
                zs.append(z.flatten(start_dim=1).cpu())
                ys.append(y.cpu())
        return torch.cat(zs), torch.cat(ys)

    train_z, train_y = extract(train_loader)
    val_z, val_y = extract(val_loader)

    probe = nn.Linear(train_z.size(1), num_classes).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = torch.utils.data.TensorDataset(train_z, train_y)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    probe.train()
    for _ in range(15):
        for z_batch, y_batch in train_dl:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            opt.zero_grad()
            logits = probe(z_batch)
            loss_fn(logits, y_batch).backward()
            opt.step()

    probe.eval()
    correct = total = 0
    with torch.no_grad():
        val_z, val_y = val_z.to(device), val_y.to(device)
        preds = probe(val_z).argmax(dim=1)
        correct += (preds == val_y).sum().item()
        total += val_y.size(0)
    probing_accuracy = correct / total

    score = recon_mse / probing_accuracy if probing_accuracy > 0 else float('inf')
    return recon_mse, probing_accuracy, score

# Hyperparameter tuning and training
def main():
    dataset = CustomDataset("train.npz", augment=True)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    # Optimized hyperparameters
    best_config = {
        'latent_channels': 8,
        'lr': 1e-4,
        'batch_size': 32,
        'alpha': 0.7,
        'beta': 1.0
    }
    input_channels = 3  # RGB images per description

    print(f"\nTraining final model with config: {best_config}")
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=best_config["batch_size"], shuffle=True, num_workers=1)
    val_loader = DataLoader(val_subset, batch_size=best_config["batch_size"], num_workers=1)

    model = Model(
        num_classes=170,
        latent_channels=best_config["latent_channels"],
        input_channels=input_channels
    ).to(device)

    best_state_dict, best_val_loss = train_model(
        model, train_loader, val_loader,
        alpha=best_config["alpha"],
        beta=best_config["beta"],
        lr=best_config["lr"],
        num_epochs=500,
    )
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), "checkpoint.pt")
    print(f"Model weights saved to checkpoint.pt, Best Val Loss: {best_val_loss:.6f}")

    def plot_reconstructions(model, dataloader, device, num_images=8):
        model.eval()
        with torch.no_grad():
            x, y = next(iter(dataloader))
            x = x.to(device)
            z = model.encode(x)
            x_recon = model.decode(z)
            x = x.cpu().numpy()
            x_recon = x_recon.cpu().numpy()
            print(f"Latent bottleneck dimension: {z.flatten(start_dim=1).shape[1]}")

            plt.figure(figsize=(16, 4))
            for i in range(num_images):
                plt.subplot(2, num_images, i+1)
                plt.imshow(x[i].transpose(1, 2, 0).squeeze(), cmap='gray' if input_channels == 1 else None)
                plt.axis('off')
                plt.subplot(2, num_images, i+1+num_images)
                plt.imshow(x_recon[i].transpose(1, 2, 0).squeeze(), cmap='gray' if input_channels == 1 else None)
                plt.axis('off')
            plt.savefig('1.png')

    plot_reconstructions(model, train_loader, device, num_images=8)

    def check_model(model_path, weights_path):
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model = model_module.Model(input_channels=input_channels, num_classes=170, latent_channels=best_config["latent_channels"])
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print("Model loaded successfully")

        test_data_tensor = torch.rand(3, input_channels, 128, 128)
        test_labels_tensor = torch.zeros(3, dtype=torch.long)
        
        model = model.to("cpu")
        model.eval()

        with torch.no_grad():
            z = model.encode(test_data_tensor)
            x_recon = model.decode(z)
        print("Model loaded successfully and encoder-decoder pipeline tested.")
        return test_data_tensor.numpy(), test_labels_tensor.numpy(), model

    test_data_numpy_from_check, test_label_numpy_from_check, loaded_model = check_model("model.py", "checkpoint.pt")

    def run_inference_AE(test_data_numpy, test_label_numpy, num_classes,
                        model_e, model_d, gpu_index,
                        batch_size=64, timeout=50, bottleNeckDim=8192):
        device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model_e.to(device)
        model_e.eval()
        model_d.to(device)
        model_d.eval()

        test_data = torch.tensor(test_data_numpy, dtype=torch.float32)
        test_labels = torch.tensor(test_label_numpy, dtype=torch.long)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        all_latents = []
        criterion = nn.MSELoss(reduction='mean')  # Align with training
        reconstruction_loss = 0
        shape_checked = False
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                images = images.to(device)
                latents = model_e.encode(images)
                
                print("latents shape:", latents.shape)
                if not shape_checked:
                    latents_orig_shape = latents.shape
                    latents = latents.view(latents.shape[0], -1)
                    if latents.shape[1] > bottleNeckDim:
                        raise ValueError(f"Latents shape is too large: {latents.shape}. Expected less than {bottleNeckDim}.")
                    latents = latents.view(latents_orig_shape)
                    shape_checked = True
                
                outputs = model_d.decode(latents)
                loss = criterion(outputs, images)
                reconstruction_loss += loss.item()
                
                all_latents.append(latents.cpu().numpy())
                    
            reconstruction_loss = reconstruction_loss / len(test_loader)
            
            all_latents = np.concatenate(all_latents, axis=0)
            mean_latents = np.mean(all_latents, axis=0)
            std_latents = np.std(all_latents, axis=0)
            
            random_latents = np.random.normal(mean_latents, std_latents, (all_latents[:5].shape))
            random_latents = torch.tensor(random_latents, dtype=torch.float32).to(device)
            sampled_images = model_d.decode(random_latents)
            sampled_images = sampled_images.cpu().numpy()
        
        torch.cuda.empty_cache()

    run_inference_AE(
        test_data_numpy=test_data_numpy_from_check,
        test_label_numpy=test_label_numpy_from_check,
        num_classes=170,
        model_e=loaded_model,
        model_d=loaded_model,
        gpu_index=0,
        batch_size=64,
        timeout=50,
        bottleNeckDim=8192
    )

if __name__ == "__main__":
    main()
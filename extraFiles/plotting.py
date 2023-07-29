import matplotlib.pyplot as plt

# Step 1: Read the .out file and extract the necessary information
file_path = 'C:/Users/Nadine/Downloads/cil_train_fine.out'

train_epochs = []
train_losses = []
train_accuracies = []
val_epochs = []
val_losses = []
val_accuracies = []

with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('Train Epoch'):
            parts = line.split()
            epoch = int(parts[2])
            train_epochs.append(epoch)

            # Extract training loss and accuracy values
            train_loss = float(parts[5])
            train_accuracy = float(parts[7])
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

        elif line.startswith('Validation Epoch'):
            parts = line.split()
            #print(parts)
            epoch = int(parts[2])
            val_epochs.append(epoch)

            # Extract validation loss and accuracy values
            val_loss = float(parts[4])
            val_accuracy = float(parts[8])
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

# Step 2: Create two separate plots for Training and Validation Loss, and Training and Validation Accuracy
plt.figure(figsize=(12, 5))

# Subplot 1: Training Loss and Validation Loss
plt.subplot(1, 2, 1)
plt.scatter(train_epochs, train_losses, s=5, label='Train Loss', color='blue')
plt.plot(val_epochs, val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Subplot 2: Training Accuracy and Validation Accuracy
plt.subplot(1, 2, 2)
plt.scatter(train_epochs, train_accuracies, s=5, label='Train Accuracy', color='green')
plt.plot(val_epochs, val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


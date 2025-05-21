import kagglehub
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torcheval import metrics
from torchvision.utils import save_image
import PIL
import random

# Seed random number generator
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Set device
#path = kagglehub.dataset_download("orvile/bone-fracture-dataset")

## Import the data from the files (must manually place in data directory)
path = "./data"
dataset = []
# !!! SET THIS EQUAL TO 1 IN ORDER TO TRAIN ON THE FULL DATASET!!!
divisor = 1
training_split = 0.7
validation_split = 0.15
test_split = 0.15

for i in range(5000):
    try:
        index = str(i)
        while len(index) < 7:
            index = "0" + index
        dataset.append([tv_tensors.Image(PIL.Image.open("./data_new/normal/IMG" + index + ".jpg")), 0])
    except:
        continue
for i in range(5000):
    try:
        index = str(i)
        while len(index) < 7:
            index = "0" + index
        dataset.append([tv_tensors.Image(PIL.Image.open("./data_new/fracture/IMG" + index + ".jpg")), 1])
    except:
        continue

random.shuffle(dataset)




# Split into training, validation, and test set
training = dataset[0:int(training_split * len(dataset))]
validation = dataset[int(training_split * len(dataset)):int(training_split * len(dataset) + validation_split * len(dataset))]
test = dataset[int(training_split * len(dataset) + validation_split * len(dataset)):]

print("Training dataset size:: ", len(training))
print("Validation dataset size:: ", len(validation))
print("Test dataset size:: ", len(test))

# Need to upsample from underrepresented class (Normal class) in the training set
num_normal = 0
num_broken = 0
for _, label in training:
    num_broken += label
    num_normal += (not label)

print("Number of broken samples: ", num_broken)
print("Number of normal samples: ", num_normal)

len_training = len(training)

while num_broken < num_normal:
    r_num = random.randint(0, len_training)
    if(training[r_num][1] == 1):
        add_sample = training[r_num]
        training.append(add_sample)
        num_broken+=1
    
random.shuffle(training)

num_normal = 0
num_broken = 0
for _, label in training:
    num_broken += label
    num_normal += (not label)

print("Number of upsampled broken samples: ", num_broken)
print("Number of upsampled normal samples: ", num_normal)

# Data augmentation
training_data_original = []
training_data_downsized = []
training_data_augmented = []

validation_data_original = []
validation_data_downsized = []

test_data_original = []
test_data_downsized = []

training_labels_original = []
training_labels_downsized = []
training_labels_augmented = []

validation_labels = []
test_labels = []


transform_base = v2.Compose([
    v2.Resize((512,512)),
    v2.Grayscale(1),
    v2.ConvertImageDtype(torch.float32)
])

transform_downsize = v2.Compose([
    v2.Resize((224,224)),
    v2.Grayscale(1),
    v2.ConvertImageDtype(torch.float32)
])

transform_aug1 = v2.Compose([
    v2.Resize((512,512)),
    v2.Grayscale(1),
    v2.ConvertImageDtype(torch.float32),
    v2.RandomVerticalFlip(1)
])

transform_aug2 = v2.Compose([
    v2.Resize((512,512)),
    v2.Grayscale(1),
    v2.ConvertImageDtype(torch.float32),
    v2.RandomHorizontalFlip(1)
])

for sample in training:
    training_data_original.append(transform_base(sample[0]))
    training_data_augmented.append(transform_base(sample[0]))
    training_data_augmented.append(transform_aug1(sample[0]))
    training_data_augmented.append(transform_aug2(sample[0]))
    training_data_downsized.append(transform_downsize(sample[0]))
    training_labels_original.append(sample[1])
    training_labels_augmented.append(sample[1])
    training_labels_augmented.append(sample[1])
    training_labels_augmented.append(sample[1])
    training_labels_downsized.append(sample[1])

for sample in validation:
    validation_data_original.append(transform_base(sample[0]))
    validation_data_downsized.append(transform_downsize(sample[0]))
    validation_labels.append(sample[1])

for sample in test:
    test_data_original.append(transform_base(sample[0]))
    test_data_downsized.append(transform_downsize(sample[0]))
    test_labels.append(sample[1])

print("Augmented training dataset size:: ", len(training_data_augmented))


# def save_examples(dataset, number, label):
#     for i in range(20, number):
#         save_image(dataset[i][0], "./output/" + label + str(i) + ".png")

# save_examples(training_data_downsized, 60, "downsized")

# Parse as custom PyTorch DataSet
class Parse_Data(Dataset):
    def __init__(self, training_data, training_labels, transform=None, target_transform=None):
        self.img_labels = training_labels
        self.imgs = training_data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label

# Put in DataLoaders
train_args = {'batch_size': 16}
val_args = {'batch_size': 16}
test_args = {'batch_size': 16}
train_loader_original = torch.utils.data.DataLoader(Parse_Data(training_data_original, training_labels_original), **train_args)
train_loader_augmented = torch.utils.data.DataLoader(Parse_Data(training_data_augmented, training_labels_augmented), **train_args)
train_loader_downsized = torch.utils.data.DataLoader(Parse_Data(training_data_downsized, training_labels_downsized), **train_args)

validation_loader_original = torch.utils.data.DataLoader(Parse_Data(validation_data_original, validation_labels), **val_args)
validation_loader_downsized = torch.utils.data.DataLoader(Parse_Data(validation_data_downsized, validation_labels), **val_args)

test_loader_original = torch.utils.data.DataLoader(Parse_Data(test_data_original, test_labels), **test_args)
test_loader_downsized = torch.utils.data.DataLoader(Parse_Data(test_data_downsized, test_labels), **test_args)

# Compute Baselines

# TRAIN AND TEST FUNCTIONS FROM PYTORCH MNIST EXAMPLE
def train(model, device, train_loader, val_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to(torch.float32)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print("OUTPUT: ", output)
            # print("TARGET: ", target)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        validate(model, device, val_loader, epoch)

# TRAIN AND TEST FUNCTIONS FROM PYTORCH MNIST EXAMPLE
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(torch.float32)
            data, target = data.to(device), target.to(device)
            targets.append(target.tolist())
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = (output>0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    #targets = torch.tensor(targets)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) #, metrics.functional.binary_auprc(output, targets)))

    
def validate(model, device, val_loader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            target = target.to(torch.float32)
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = (output>0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nTrain Epoch: {} \tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
# Baseline: guess fractured
class BaseLine_Static(nn.Module):
    def __init__(self):
        super(BaseLine_Static, self).__init__()

    def forward(self, x):
        output = torch.ones((1,), dtype=torch.float32, device=x.device)
        return output.new_full((x.shape[0], ),1.0)

class Our_Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k_size = kwargs['k_size']
        self.k_stride = kwargs['k_stride']
        self.k_pad = kwargs['k_pad']
        self.p_size = kwargs['p_size']
        self.p_stride = kwargs['p_stride']
        self.p_pad = kwargs['p_pad']
        
        self.cnn_block = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.p_size, stride=self.p_stride, padding=self.p_pad),
            nn.Conv2d(32, 64, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.p_size, stride=self.p_stride, padding=self.p_pad),
            nn.Conv2d(64, 128, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.flat_dim = 128 * self.calc_dim(kwargs)  # Multiplies num_channels and 2d_params to get total parameters for a single sample
        
        self.ffn_block = torch.nn.Sequential(
            nn.Linear(self.flat_dim, 1)
        )
    
    def forward(self, x):
        cnn_out = self.cnn_block(x)
        flattened = cnn_out.view(-1, self.flat_dim)
        lin_out = self.ffn_block(flattened)
        # print("RAW PRED: ", lin_out)
        pred = nn.functional.sigmoid(lin_out)
        return pred.view(-1)

    def calc_dim(self, kwargs):
        # Can literally just pass a 1x1x224x244 tensor through and check final dimensions
        H, W = kwargs['i_size'], kwargs['i_size']
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        H, W = (H + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1, (W + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        H, W = (H + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1, (W + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        # dummy = torch.rand([1, 1, 224, 224], dtype=torch.float32).to(device)
        # # print(dummy.shape)
        # dummy = self.cnn_block(dummy)
        # params_2d = dummy.shape[2] * dummy.shape[3]
        return int(H)*int(W)
    
baseline_model1 = BaseLine_Static()

#Set convolutional parameters
k_size = 3
k_stride = 1
k_pad = 1
p_size = 2
p_stride = 2
p_pad = 0
i_size = 512

epochs = 5  # Set epochs for training

# Baseline
baseline_model = BaseLine_Static()

# Instantiate models
model = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
model_augmented = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
i_size = 224
model_downsized = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.2)
optimizer_augmented = torch.optim.SGD(model_augmented.parameters(), lr=0.0001, weight_decay=0.2)
optimizer_downsized = torch.optim.SGD(model_downsized.parameters(), lr=0.0001, weight_decay=0.2)


# Train for set number of epochs
print("!!!!!!! TRAINING STANDARD DATASET !!!!!!!!")
train(model, device, train_loader_original, validation_loader_original, optimizer, epochs)
print("!!!!!!! TRAINING AUGMENTED DATASET !!!!!!!!")
train(model_augmented, device, train_loader_augmented, validation_loader_original, optimizer_augmented, epochs)
print("!!!!!!! TRAINING DOWNSIZED DATASET !!!!!!!!")
train(model_downsized, device, train_loader_downsized, validation_loader_downsized, optimizer_downsized, epochs)

# Test trained model
print("!!!!!!! TESTING STANDARD DATASET !!!!!!!!")
test(model, device, test_loader_original)
print("!!!!!!! TESTING AUGMENTED DATASET !!!!!!!!")
test(model_augmented, device, test_loader_original)
print("!!!!!!! TESTING DOWNSIZED DATASET !!!!!!!!")
test(model_downsized, device, test_loader_downsized)

# Test baseline model
print("!!!!!!! TESTING BASELINE !!!!!!!!")
test(baseline_model, device, test_loader_original)


torch.save(model.state_dict(), './standard_model_dataset2.pt')
print('Standard Model Saved')
torch.save(model_augmented.state_dict(), './augmented_model_dataset2.pt')
print('Augmented Model Saved')
torch.save(model_downsized.state_dict(), './downsized_model_dataset2.pt')
print('Downsized Model Saved')

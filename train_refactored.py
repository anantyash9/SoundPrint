from multiprocessing import cpu_count
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from sklearn.model_selection import train_test_split
from olympic.callbacks import Evaluate, ReduceLROnPlateau, ModelCheckpoint
from olympic import fit
from soundprint.datasets import LibriSpeech
from soundprint.models import ResidualClassifier
from soundprint.utils import whiten, setup_dirs
from soundprint.eval import VerificationMetrics
from config import PATH

# Set up foders and GPU
setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


# Traning Parameters
n_seconds = 3
downsampling = 4
batch_size = 100
filters = 4
epochs = 100
val_fraction = 0.1
in_channels = 1
model_file_name = '__'.join(str(d) for d in [
                            n_seconds, downsampling, batch_size, filters, epochs, in_channels])


# Split Dataset

librispeech = LibriSpeech(['train-clean-100', 'train-clean-360', 'train-other-500'], n_seconds,
                          downsampling, stochastic=True, pad=False)
librispeech_unseen = LibriSpeech(
    'dev-clean', n_seconds, downsampling, stochastic=True, pad=False)
data = librispeech
num_classes = data.num_classes
print(f'Distinct Speakers= {num_classes}')

indices = range(len(data))
train_indices, test_indices, _, _ = train_test_split(
    indices,
    indices,
    test_size=val_fraction,
)

train = torch.utils.data.Subset(data, train_indices)
val = torch.utils.data.Subset(data, test_indices)


# Create Model Instance
model = ResidualClassifier(in_channels, filters, [
                           2, 2, 2, 2], num_classes, dim=in_channels)
model.to(device, dtype=torch.double)


# Create Dataloader Instances
train_loader = DataLoader(train, batch_size=batch_size,
                          num_workers=cpu_count(), shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size,
                        num_workers=cpu_count(), shuffle=True, drop_last=False)

#Set optimizer and loss function
opt = optim.Adadelta(model.parameters())
loss_fn = nn.CrossEntropyLoss()


def prepare_batch(batch):
    x, y = batch
    return whiten(x).cuda(), y.long().cuda()

def gradient_step(model, optimiser, loss_fn, x, y, epoch):
    model.train()
    optimiser.zero_grad()
    y_pred = model(x, y)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred

#Callbacks & Checkpoint savers
callbacks = [
    Evaluate(
        DataLoader(
            train,
            num_workers=cpu_count(),
            batch_sampler=BatchSampler(RandomSampler(
                train, replacement=True, num_samples=20000), batch_size, True)
        ),
        prefix='train_'
    ),
    Evaluate(val_loader),
    VerificationMetrics(librispeech_unseen, num_pairs=20000,
                        prefix='librispeech_dev_clean_'),
    ReduceLROnPlateau(monitor='val_loss', patience=5,
                      verbose=True, min_delta=0.25),
    ModelCheckpoint(filepath=PATH + f'/models/{model_file_name}.pt',
                    monitor='val_loss', save_best_only=True, verbose=True),
]

fit(
    model,
    opt,
    loss_fn,
    epochs=epochs,
    dataloader=train_loader,
    prepare_batch=prepare_batch,
    callbacks=callbacks,
    metrics=['accuracy'],
    update_fn=gradient_step
)

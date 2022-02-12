import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 99


class ModelToolkit:

    def __init__(self, model, name, checkpoint=None):
        self.model = model
        self.name = name
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.scheduler = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=5,
                                                                    verbose=True)
        self.epoch = 0
        self.best_loss = float('inf')

        if checkpoint is not None:
            self.load_model(checkpoint)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(self.device, torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')
            print(self.device)
        self.model = self.model.to(self.device)

    def save_model(self, description=''):
        file_name = '{}_e{}({})'.format(self.name, self.epoch, description)
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimazer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, 'checkpoints/{}'.format(file_name))
        print('saving model with name: "{}"'.format(file_name))
        return file_name

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimazer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        print('loading model with name: "{}"'.format(file_name))

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def predict(self, images):
        images = images.to(self.device)
        outputs = self.model(images)
        outputs = outputs.detach().cpu()
        return outputs

    def train(self, train_dl, val_dl, num_epochs):
        writer = SummaryWriter(logdir=os.path.join('runs', self.name))
        for epoch in range(num_epochs):
            self.epoch += 1
            writer.add_scalar('train loss', self.run_epoch('train', train_dl), self.epoch)

            with torch.no_grad():
                val_loss = self.run_epoch('val', val_dl)
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                writer.add_scalar('val loss', val_loss, self.epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('******** New optimal found, saving state ********')
                best_model_name = self.save_model(description='{:.4f}'.format(val_loss))
        self.save_model(description='{:.4f}'.format(val_loss))
        writer.close()
        return best_model_name

    def run_epoch(self, phase, dataloader):
        start = time.strftime('%H:%M:%S')
        print(f'Starting epoch: {self.epoch} | phase: {phase} | ⏰: {start}')
        self.model.train(phase == 'train')
        running_loss = 0.0
        tk = tqdm(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            tk.set_postfix({'loss': (running_loss / (itr + 1))})
        return running_loss / len(dataloader)


class FaceBoxDataset(Dataset):
    def __init__(self, df, phase='val'):
        self.df = df
        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

    def __getitem__(self, i):
        f = self.df.index[i]
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[self.df.loc[f, 'ymin']:self.df.loc[f, 'ymax'], self.df.loc[f, 'xmin']:self.df.loc[f, 'xmax']]
        xs = (self.df.loc[f, 'x1':'x68'].to_numpy() - self.df.loc[f, 'xmin'])
        ys = (self.df.loc[f, 'y1':'y68'].to_numpy() - self.df.loc[f, 'ymin'])
        keypoints = []
        for x, y in zip(xs, ys):
            keypoints.append((x, y))
        transformed = self.transforms(image=img, keypoints=keypoints)
        xs = []
        ys = []
        for (x, y) in transformed['keypoints']:
            xs.append(x)
            ys.append(y)
        return transformed['image']/255, torch.tensor(xs + ys, dtype=torch.float)

    def __len__(self):
        return len(self.df.index)


def get_dataloaders(df, batch_size=8, num_workers=12, test_split=0.2):
    train_df, val_df = train_test_split(df, test_size=test_split)
    train_dataset = FaceBoxDataset(train_df, phase='train')
    val_dataset = FaceBoxDataset(val_df, phase='val')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  )
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                )
    return train_dataloader, val_dataloader


def test_with_ced(model, df):
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])

    dist_err = []
    mse_err = []
    for f in tqdm(df.index):
        img = cv2.imread(f)
        img = img[df.loc[f, 'ymin']:df.loc[f, 'ymax'], df.loc[f, 'xmin']:df.loc[f, 'xmax']]
        img = test_transforms(img)
        preds = model.predict(img.unsqueeze(0))[0].numpy()
        width = (df.loc[f, 'xmax'] - df.loc[f, 'xmin'])
        height = (df.loc[f, 'ymax'] - df.loc[f, 'ymin'])
        x_pred = preds[0:68] * width / IMG_SIZE + df.loc[f, 'xmin']
        y_pred = preds[68:] * height / IMG_SIZE + df.loc[f, 'ymin']
        dist_err.append(count_dist_err(
            x_pred,
            y_pred,
            df.loc[f, 'x1':'x68'].to_numpy(),
            df.loc[f, 'y1':'y68'].to_numpy()
        ))
        mse_err.append(count_mse_err(
            np.concatenate((x_pred, y_pred)),
            df.loc[f, 'x1':'y68'].to_numpy(),
            np.sqrt((df.loc[f, 'bottom'] - df.loc[f, 'top']) * (df.loc[f, 'right'] - df.loc[f, 'left']))
        ))
    dist_err = np.sort(dist_err)
    mse_err = np.sort(mse_err)
    path = df.index[0].split('/')
    return mse_err, dist_err
    # draw_ced(dist_err, '{}_dist_square_{}_{}'.format(model.name, path[1], path[2]))
    # draw_ced(mse_err, '{}_mse_square_{}_{}'.format(model.name, path[1], path[2]))
    # draw_ced(mse_err, '{}_mse_square_8_{}_{}'.format(model.name, path[1], path[2]), stop=0.8)
    # draw_ced(dist_err, '{}_dist_square_8_{}_{}'.format(model.name, path[1], path[2]), stop=0.8)


def count_dist_err(x_pred, y_pred, x_gt, y_gt):
    n_points = x_pred.shape[0]
    assert n_points == x_gt.shape[0], '{} != {}'.format(n_points, x_gt.shape[0])

    w = np.max(x_gt) - np.min(x_gt)
    h = np.max(y_gt) - np.min(y_gt)
    normalization_factor = np.sqrt(h * w)

    diff_x = x_gt - x_pred
    diff_y = y_gt - y_pred
    dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
    avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
    return avg_norm_dist


def count_mse_err(preds, targets, norm_factor):
    assert preds.shape[0] == targets.shape[0], '{} != {}'.format(preds.shape[0], targets.shape[0])
    return ((targets - preds) ** 2).mean() / norm_factor


def draw_ced(errors, names, output_name, stop=0.08):
    plt.figure(figsize=(30, 20), dpi=100)
    plt.title('CED')
    colors = ['b', 'g', 'r', 'c']
    for i, (error, name) in enumerate(zip(errors, names)):
        error = np.array(error)
        percents = []
        auc = 0
        thresholds = []
        step = stop / 100
        for thr in np.arange(0.0, stop + step / 2, step):
            percent = np.count_nonzero(error < thr) / error.shape[0]
            thresholds.append(thr)
            percents.append(percent)
            auc += percent * step
        plt.plot(thresholds, percents, color=colors[i], label='{} auc={:1.3f}'.format(name, auc), linewidth=3.0)
    plt.legend(loc='lower right', fontsize='xx-large')
    if not os.path.exists('CED'):
        os.mkdir('CED')
    plt.savefig(os.path.join('CED', output_name))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import kaggle

# 1. Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используется устройство: {device}')

# 2. Настройка Kaggle API
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
kaggle_json_path = os.path.join(os.path.dirname(__file__), 'kaggle.json')
if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
    import shutil
    shutil.copy(kaggle_json_path, os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# 3. Скачивание датасета
if not os.path.exists('vangogh2photo'):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('srrrrr/vangogh2photo', path='.', unzip=True)

# 4. Класс датасета
class VanGoghDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 5. Преобразования и загрузчики
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
vangogh_dataset = VanGoghDataset('vangogh2photo/trainA', transform=transform)
photo_dataset = VanGoghDataset('vangogh2photo/trainB', transform=transform)
vangogh_subset = Subset(vangogh_dataset, range(min(200, len(vangogh_dataset))))
photo_subset = Subset(photo_dataset, range(min(200, len(photo_dataset))))
vangogh_loader = DataLoader(vangogh_subset, batch_size=1, shuffle=True, num_workers=2)
photo_loader = DataLoader(photo_subset, batch_size=1, shuffle=True, num_workers=2)
print(f'Используем {len(vangogh_subset)} картин Ван Гога и {len(photo_subset)} фотографий')

# 6. Модели
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, img):
        return self.model(img)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

# --- ДОБАВЛЕНИЕ: Продолжение обучения с чекпоинта ---
def continue_training(G_B2A, G_A2B, D_A, D_B, vangogh_loader, photo_loader, num_epochs=30):
    optimizer_G = optim.Adam(list(G_B2A.parameters()) + list(G_A2B.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion_GAN = GANLoss().to(device)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(vangogh_loader, photo_loader)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            optimizer_G.zero_grad()
            same_B = G_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            same_A = G_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0
            fake_A = G_B2A(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, True)
            fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, True)
            recov_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B) * 10.0
            recov_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * 10.0
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_B2A + loss_GAN_A2B + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            optimizer_G.step()
            optimizer_D_A.zero_grad()
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, True)
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            optimizer_D_B.zero_grad()
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, True)
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            if i % 50 == 0:
                print(f'[Continue] Epoch [{epoch+1}/{num_epochs}], Step [{i}], '
                      f'Loss_D: {loss_D_A.item():.4f}, {loss_D_B.item():.4f}, '
                      f'Loss_G: {loss_G.item():.4f}')
    torch.save(G_B2A.state_dict(), 'generator_B2A.pth')
    print('Веса генератора G_B2A сохранены после дообучения!')
    return G_B2A

# --- ДОБАВЛЕНИЕ: Суперразрешение (Real-ESRGAN) ---
def super_resolve(input_path, output_path, scale=4):
    try:
        from realesrgan import RealESRGAN
        from PIL import Image
        model = RealESRGAN(device, scale=scale)
        model.load_weights('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4.pth')
        img = Image.open(input_path).convert('RGB')
        sr_img = model.predict(img)
        sr_img.save(output_path)
        print(f'Super-resolved image saved to {output_path}')
    except ImportError:
        print('Установите пакет realesrgan: pip install realesrgan')

# 7. Функция обучения

def train_cyclegan():
    G_B2A = Generator().to(device)
    G_A2B = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    optimizer_G = optim.Adam(list(G_B2A.parameters()) + list(G_A2B.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion_GAN = GANLoss().to(device)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(vangogh_loader, photo_loader)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            optimizer_G.zero_grad()
            same_B = G_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            same_A = G_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0
            fake_A = G_B2A(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, True)
            fake_B = G_A2B(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, True)
            recov_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B) * 10.0
            recov_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * 10.0
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_B2A + loss_GAN_A2B + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            optimizer_G.step()
            optimizer_D_A.zero_grad()
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, True)
            pred_fake = D_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            optimizer_D_B.zero_grad()
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, True)
            pred_fake = D_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/30], Step [{i}], '
                      f'Loss_D: {loss_D_A.item():.4f}, {loss_D_B.item():.4f}, '
                      f'Loss_G: {loss_G.item():.4f}')
    return G_B2A, G_A2B

# 8. Запуск обучения и сохранение модели
if __name__ == '__main__':
    # Если есть сохранённые веса, загрузим их
    if os.path.exists('generator_B2A.pth'):
        print('Загружаю веса генератора G_B2A для дообучения...')
        G_B2A = Generator().to(device)
        G_B2A.load_state_dict(torch.load('generator_B2A.pth', map_location=device))
    else:
        G_B2A = Generator().to(device)
    G_A2B = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Дообучение с чекпоинта
    G_B2A = continue_training(G_B2A, G_A2B, D_A, D_B, vangogh_loader, photo_loader, num_epochs=30)
    # Пример суперразрешения (после генерации изображения)
    # super_resolve('results/your_image.png', 'results/your_image_sr.png', scale=4)

    torch.save(G_B2A.state_dict(), 'generator_B2A.pth')
    print('Модель генератора (фото -> Ван Гог) сохранена в generator_B2A.pth') 
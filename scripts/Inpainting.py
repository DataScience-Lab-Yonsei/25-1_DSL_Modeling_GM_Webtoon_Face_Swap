import os
import glob
import pickle
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("scikit-learn이 필요합니다. 'pip install scikit-learn' 명령어로 설치하세요.")


## ------------------ Data Preprocessor ------------------ ##
class DataPreprocessor:
    def __init__(self, final_size=1024):
        self.final_size = final_size

    def center_pad_to(self, img):
        w, h = img.size
        if w > self.final_size or h > self.final_size:
            img = img.resize((self.final_size, self.final_size), Image.BICUBIC)
            w, h = img.size
        new_img = Image.new("RGB", (self.final_size, self.final_size), (0, 0, 0))
        left = (self.final_size - w) // 2
        top = (self.final_size - h) // 2
        new_img.paste(img, (left, top))
        return new_img, left, top, w, h

    def prepare_presized_data(self, original_files, root_dir, presized_root, padding_info_path):
        """
        이미지들을 1024x1024 기준으로 중앙 패딩 및 리사이즈 후, presized_root에 저장.
        padding_info는 {상대경로: (left, top, w, h)} 형태로 pickle에 저장.
        """
        if os.path.exists(presized_root):
            print(f"[Info] 이미 '{presized_root}' 폴더가 존재합니다. 기존 데이터를 사용합니다.")
            if os.path.exists(padding_info_path):
                with open(padding_info_path, "rb") as f:
                    padding_info = pickle.load(f)
                print(f"[Info] 기존 padding_info {padding_info_path} 로드 완료.")
                return padding_info
            else:
                print("[Warning] padding_info.pkl이 없어 패딩 범위 정보를 활용할 수 없습니다.")
                return {}
        else:
            print(f"[Info] '{presized_root}' 폴더가 없으므로 새로 생성합니다.")
            os.makedirs(presized_root, exist_ok=True)

        padding_info = {}
        print("[Info] 1024x1024에 맞는 중앙 패딩된 이미지를 생성 중...")

        for rel_path in tqdm(original_files):
            src_path = os.path.join(root_dir, rel_path)
            dst_path = os.path.join(presized_root, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            img = Image.open(src_path).convert('RGB')
            img_1024, left, top, w, h = self.center_pad_to(img)
            img_1024.save(dst_path)

            padding_info[rel_path] = (left, top, w, h)

        with open(padding_info_path, "wb") as f:
            pickle.dump(padding_info, f)
        print(f"[Info] padding_info를 {padding_info_path}에 저장했습니다.")

        return padding_info


## ------------------ Dataset ------------------ ##
class InpaintDataset(Dataset):
    """
    (Train 모드) mask를 랜덤 생성
    """
    def __init__(self, file_list, root_dir, padding_info, final_size=1024,
                 mode='train', min_size_threshold=128):
        super().__init__()
        self.root_dir = root_dir
        self.final_size = final_size
        self.mode = mode
        self.transform = transforms.ToTensor()
        self.padding_info = padding_info if padding_info is not None else {}

        filtered_list = []
        for rel_path in file_list:
            if rel_path in self.padding_info:
                left, top, w, h = self.padding_info[rel_path]
                # 너무 작은 경우 제외
                if min(w, h) < min_size_threshold:
                    continue
            filtered_list.append(rel_path)
        self.file_list = filtered_list
        print(f"[InpaintDataset] 원래 {len(file_list)}개 중 {len(self.file_list)}개 사용 (min_size>={min_size_threshold}).")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path = self.file_list[idx]
        path = os.path.join(self.root_dir, rel_path)
        img = Image.open(path).convert('RGB')
        img_t = self.transform(img)

        if self.mode == 'train':
            left, top, w, h = self.padding_info.get(rel_path, (0, 0, self.final_size, self.final_size))
            mask_np = self._random_mask_in_valid_area(left, top, w, h)
        else:
            # val/test 모드에선 mask 없이 0
            mask_np = np.zeros((self.final_size, self.final_size), dtype=np.float32)

        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)
        return img_t, mask_t

    def _random_mask_in_valid_area(self, left, top, w, h):
        if random.random() < 0.5:
            return self._random_bbox_mask(left, top, w, h)
        else:
            return self._random_free_form_mask(left, top, w, h)

    def _random_bbox_mask(self, left, top, w, h):
        mask = np.zeros((self.final_size, self.final_size), dtype=np.float32)
        min_box = 16
        max_size = min(w, h)//2
        if max_size < min_box:
            return mask
        box_size = random.randint(min_box, max_size)
        max_left = left + (w - box_size)
        max_top = top + (h - box_size)
        if max_left < left or max_top < top:
            return mask
        box_left = random.randint(left, max_left)
        box_top = random.randint(top, max_top)
        mask[box_top:box_top+box_size, box_left:box_left+box_size] = 1.0
        return mask

    def _random_free_form_mask(self, left, top, w, h,
                               max_strokes=10, max_vertex=5,
                               max_length=40, max_brush_width=20):
        mask = np.zeros((self.final_size, self.final_size), np.uint8)
        num_strokes = np.random.randint(1, max_strokes+1)
        for _ in range(num_strokes):
            start_x = np.random.randint(left, left + w)
            start_y = np.random.randint(top, top + h)
            brush_width = np.random.randint(5, max_brush_width+1)
            num_vertex = np.random.randint(1, max_vertex+1)

            points = [(start_x, start_y)]
            for _v in range(num_vertex):
                angle = np.random.randint(0, 360)
                length = np.random.randint(10, max_length+1)
                new_x = points[-1][0] + int(length * np.cos(angle))
                new_y = points[-1][1] + int(length * np.sin(angle))
                new_x = np.clip(new_x, left, left + w - 1)
                new_y = np.clip(new_y, top, top + h - 1)
                points.append((new_x, new_y))

            points = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [points], isClosed=False, color=255, thickness=brush_width)
            cv2.circle(mask, (points[-1][0], points[-1][1]), brush_width//2, 255, -1)

        mask = mask.astype(np.float32) / 255.
        return mask


## ------------------ Network ------------------ ##
class UNetGenerator(nn.Module):
    """
    간단한 U-Net Generator (RGB+mask -> RGB)
    """
    def __init__(self, in_ch=4, out_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = self._block(in_ch, base_ch, norm=False)
        self.enc2 = self._block(base_ch, base_ch*2)
        self.enc3 = self._block(base_ch*2, base_ch*4)
        self.enc4 = self._block(base_ch*4, base_ch*8)
        self.enc5 = self._block(base_ch*8, base_ch*8)

        self.dec4 = self._block(base_ch*8 + base_ch*8, base_ch*8)
        self.dec3 = self._block(base_ch*8 + base_ch*4, base_ch*4)
        self.dec2 = self._block(base_ch*4 + base_ch*2, base_ch*2)
        self.dec1 = self._block(base_ch*2 + base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

    def _block(self, in_c, out_c, norm=True):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(True)
        ]
        if norm:
            layers.insert(1, nn.BatchNorm2d(out_c))
            layers.append(nn.BatchNorm2d(out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        e5 = self.enc5(p4)

        up4 = F.interpolate(e5, scale_factor=2, mode='nearest')
        d4 = torch.cat([up4, e4], dim=1)
        d4 = self.dec4(d4)

        up3 = F.interpolate(d4, scale_factor=2, mode='nearest')
        d3 = torch.cat([up3, e3], dim=1)
        d3 = self.dec3(d3)

        up2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = torch.cat([up2, e2], dim=1)
        d2 = self.dec2(d2)

        up1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = torch.cat([up1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.outc(d1)
        return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_ch, base_ch, 4, 2, 1),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
                   nn.BatchNorm2d(base_ch*2),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
                   nn.BatchNorm2d(base_ch*4),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(base_ch*4, base_ch*8, 4, 1, 1),
                   nn.BatchNorm2d(base_ch*8),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(base_ch*8, 1, 4, 1, 1)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


## ------------------ Loss Functions ------------------ ##
def gan_loss(pred, is_real=True, use_lsgan=False):
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    if use_lsgan:
        return F.mse_loss(pred, target)
    else:
        return F.binary_cross_entropy_with_logits(pred, target)


def masked_l1_loss(pred, gt, mask):
    return F.l1_loss(pred*mask, gt*mask)


def total_variation_loss(x, tv_weight=1e-5):
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_weight * (dh + dw)


def boundary_enhancement_loss(pred, gt):
    """
    라플라시안 필터를 통한 경계 강조
    """
    pred_gray = torch.mean(pred, dim=1, keepdim=True)
    gt_gray = torch.mean(gt, dim=1, keepdim=True)
    lap_filter = torch.tensor([[0., -1., 0.],
                               [-1., 4., -1.],
                               [0., -1., 0.]], device=pred.device).view(1, 1, 3, 3)
    pred_lap = F.conv2d(pred_gray, lap_filter, padding=1)
    gt_lap = F.conv2d(gt_gray, lap_filter, padding=1)
    return F.l1_loss(pred_lap, gt_lap)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], use_l1=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        prev_id = 0
        for i in layer_ids:
            slice_i = nn.Sequential(*vgg[prev_id:i])
            self.slices.append(slice_i)
            prev_id = i

        for param in self.parameters():
            param.requires_grad = False
        self.use_l1 = use_l1

    def forward(self, x, y):
        loss = 0.0
        for slice_i in self.slices:
            x = slice_i(x)
            y = slice_i(y)
            if self.use_l1:
                loss += F.l1_loss(x, y)
            else:
                loss += F.mse_loss(x, y)
        return loss


## ------------------ Trainer ------------------ ##
class InpaintingTrainer:
    def __init__(self, netG, netD, device='cuda'):
        self.netG = netG.to(device)
        self.netD = netD.to(device)
        self.device = device

    def train_inpainting(self,
                         train_loader,
                         num_epochs=10,
                         lr=2e-4,
                         lambda_l1=1.0,
                         lambda_adv=0.05,
                         lambda_tv=1e-4,
                         use_lsgan=False,
                         use_perceptual_loss=False,
                         perceptual_model=None,
                         lambda_percep=0.05,
                         use_boundary_loss=False,
                         lambda_boundary=0.05,
                         checkpoint_path='settings/inpainting_checkpoint.pth',
                         sample_dir='train_samples'):
        """
        실제 학습 실행
        """
        os.makedirs(sample_dir, exist_ok=True)

        optG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))

        start_epoch = 0
        # 이전 체크포인트 로드
        if os.path.exists(checkpoint_path):
            print(f"-> 체크포인트 로드: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.netG.load_state_dict(ckpt['netG'])
            self.netD.load_state_dict(ckpt['netD'])
            optG.load_state_dict(ckpt['optG'])
            optD.load_state_dict(ckpt['optD'])
            start_epoch = ckpt['epoch'] + 1
            print(f"   [재시작 에포크: {start_epoch}]")

        for epoch in range(start_epoch, num_epochs):
            self.netG.train()
            self.netD.train()
            run_g, run_d = 0.0, 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for img, mask in pbar:
                img = img.to(self.device)
                mask = mask.to(self.device)

                # ---- Discriminator ---- #
                g_in = torch.cat([img, mask], dim=1)
                out = self.netG(g_in)
                completed = img*(1 - mask) + out*mask

                optD.zero_grad()
                pred_real = self.netD(img)
                d_loss_real = gan_loss(pred_real, True, use_lsgan)
                pred_fake = self.netD(completed.detach())
                d_loss_fake = gan_loss(pred_fake, False, use_lsgan)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                optD.step()

                # ---- Generator ---- #
                optG.zero_grad()
                g_l1 = masked_l1_loss(out, img, mask) * lambda_l1
                pred_fake_g = self.netD(completed)
                g_adv = gan_loss(pred_fake_g, True, use_lsgan) * lambda_adv

                g_perc = 0.0
                if use_perceptual_loss and (perceptual_model is not None):
                    g_perc = perceptual_model(completed, img) * lambda_percep

                g_tv = 0.0
                if lambda_tv > 0:
                    g_tv = total_variation_loss(completed, lambda_tv)

                g_bound = 0.0
                if use_boundary_loss:
                    g_bound = boundary_enhancement_loss(completed, img) * lambda_boundary

                g_total = g_l1 + g_adv + g_perc + g_tv + g_bound
                g_total.backward()
                optG.step()

                run_g += g_total.item()
                run_d += d_loss.item()
                pbar.set_postfix({
                    "G_loss": f"{g_total.item():.4f}",
                    "D_loss": f"{d_loss.item():.4f}"
                })

            mean_g = run_g / len(train_loader)
            mean_d = run_d / len(train_loader)
            print(f"[Epoch {epoch+1}/{num_epochs}] G_loss: {mean_g:.4f}, D_loss: {mean_d:.4f}")

            # 체크포인트 저장
            ckpt = {
                'epoch': epoch,
                'netG': self.netG.state_dict(),
                'netD': self.netD.state_dict(),
                'optG': optG.state_dict(),
                'optD': optD.state_dict()
            }
            torch.save(ckpt, checkpoint_path)
            print(f"Epoch {epoch+1} checkpoint saved -> {checkpoint_path}")


## ------------------ Sliding Inference ------------------ ##
def sliding_inference_image(img_tensor, netG, device='cuda', box_size=256, overlap_ratio=0.5):
    """
    큰 이미지를 슬라이딩 윈도우 방식으로 마스크 영역을 채워나가는 테스트용 함수
    """
    netG.eval()
    B, C, H, W = img_tensor.shape
    accum_result = img_tensor.clone().to(device)

    step = int(box_size * (1 - overlap_ratio))
    step = max(step, 1)

    top_pos = 0
    while top_pos < H:
        if top_pos + box_size > H:
            top_pos = H - box_size
        left_pos = 0
        while left_pos < W:
            if left_pos + box_size > W:
                left_pos = W - box_size

            box_bottom = top_pos + box_size
            box_right = left_pos + box_size
            if (box_bottom > H) or (box_right > W) or (box_size <= 0):
                break

            mask = torch.zeros((B, 1, H, W), device=device)
            mask[:, :, top_pos:box_bottom, left_pos:box_right] = 1.0

            with torch.no_grad():
                g_in = torch.cat([accum_result, mask], dim=1)
                out_patch = netG(g_in)
                completed_patch = accum_result * (1 - mask) + out_patch * mask

            accum_result = accum_result * (1 - mask) + completed_patch * mask

            left_pos += step
            if left_pos + box_size > W and left_pos < W:
                left_pos = W - box_size
        top_pos += step
        if top_pos + box_size > H and top_pos < H:
            top_pos = H - box_size

    return accum_result


## ------------------ Tester ------------------ ##
class InpaintingTester:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.netG = UNetGenerator(in_ch=4, out_ch=3, base_ch=64).to(device)

        print(f"[InpaintingTester] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.netG.load_state_dict(checkpoint['netG'])
        self.netG.eval()

    def test_on_folder(self, test_folder, output_folder,
                       box_size=256, overlap_ratio=0.5, target_size=1024):
        os.makedirs(output_folder, exist_ok=True)

        exts = ['*.png', '*.jpg', '*.jpeg']
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(test_folder, ext))
        files = sorted(files)
        print(f"[InpaintingTester] Found {len(files)} images in {test_folder}")

        for img_path in tqdm(files, desc='Folder Inference'):
            img_name = os.path.basename(img_path)
            out_path = os.path.join(output_folder, img_name)

            # 원본 이미지 로드
            img = Image.open(img_path).convert('RGB')
            padded_img, left, top, w, h = self.center_pad_to(img, target_size)

            img_np = np.array(padded_img, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # (C,H,W)
            img_t = torch.from_numpy(img_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                result_t = sliding_inference_image(
                    img_t, self.netG, device=self.device,
                    box_size=box_size, overlap_ratio=overlap_ratio
                )

            result_np = result_t[0].cpu().numpy()
            result_np = np.clip(result_np, 0, 1)
            result_np = np.transpose(result_np, (1, 2, 0))  # (H,W,C)

            # 원본 크기로 잘라내기
            cropped_np = result_np[top:top+h, left:left+w, :]
            cropped_np = (cropped_np * 255).astype(np.uint8)
            out_img = Image.fromarray(cropped_np)
            out_img.save(out_path)

    def center_pad_to(self, img, target_size=1024):
        w, h = img.size
        # 이미지가 target_size보다 크다면 먼저 리사이즈
        if w > target_size or h > target_size:
            img = img.resize((target_size, target_size), Image.BICUBIC)
            w, h = img.size
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        left = (target_size - w) // 2
        top = (target_size - h) // 2
        new_img.paste(img, (left, top))
        return new_img, left, top, w, h


## ------------------ Final API: train_inpainting, test_inpainting ------------------ ##
def train_inpainting(
    folder_number=112,
    num_epochs=10,
    batch_size=2,
    use_boundary_loss=True,
    lambda_boundary=0.1,
    use_perceptual_loss=False,
    lambda_percep=0.05,
    lr=2e-4,
    lambda_l1=1.0,
    lambda_adv=0.05,
    lambda_tv=1e-4,
    use_lsgan=False,
    min_size_threshold=128
):
    """
    folder_number: (기본값 112)
       results/{folder_number}/temporary_crops 폴더에서 이미지를 받아 학습.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 자동 경로 설정
    root_dir          = f"results/{folder_number}/temporary_crops"
    presized_root     = f"results/{folder_number}/presized_tem_cropped_faces"
    padding_info_path = f"results/{folder_number}/presized_tem_padding_info.pkl"
    checkpoint_path   = "settings/inpainting_checkpoint.pth"
    sample_dir        = f"results/{folder_number}/train_samples"

    # 1) 이미지 목록
    all_files = []
    for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        all_files += glob.glob(os.path.join(root_dir, ext), recursive=True)
    all_files_rel = [os.path.relpath(f, root_dir) for f in all_files]

    # 2) 1024x1024 presized 데이터 준비
    preprocessor = DataPreprocessor(final_size=1024)
    padding_info = preprocessor.prepare_presized_data(
        original_files=all_files_rel,
        root_dir=root_dir,
        presized_root=presized_root,
        padding_info_path=padding_info_path
    )

    # 3) train/val split
    train_files, val_files = train_test_split(all_files_rel, test_size=0.2, random_state=42)

    train_ds = InpaintDataset(train_files,
                              root_dir=presized_root,
                              padding_info=padding_info,
                              final_size=1024,
                              mode='train',
                              min_size_threshold=min_size_threshold)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    # 4) 모델 선언
    netG = UNetGenerator(in_ch=4, out_ch=3, base_ch=64)
    netD = PatchDiscriminator(in_ch=3, base_ch=64)
    trainer = InpaintingTrainer(netG, netD, device=device)

    # Perceptual Loss가 필요하면 준비
    perceptual_model = None
    if use_perceptual_loss:
        perceptual_model = VGGPerceptualLoss()

    # 5) 학습 수행
    trainer.train_inpainting(
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=lr,
        lambda_l1=lambda_l1,
        lambda_adv=lambda_adv,
        lambda_tv=lambda_tv,
        use_lsgan=use_lsgan,
        use_perceptual_loss=use_perceptual_loss,
        perceptual_model=perceptual_model,
        lambda_percep=lambda_percep,
        use_boundary_loss=use_boundary_loss,
        lambda_boundary=lambda_boundary,
        checkpoint_path=checkpoint_path,
        sample_dir=sample_dir
    )
    print(f"Done Training on folder_number={folder_number}!")


def test_inpainting(
    folder_number=112,
    box_size=128,
    overlap_ratio=0.5,
    target_size=1024
):
    """
    folder_number: (기본값 112)
      results/{folder_number}/face_swapped_images 폴더 내 사진을
      Inpainting 후 results/{folder_number}/final_result 에 저장.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "settings/inpainting_checkpoint.pth"

    input_folder  = f"results/{folder_number}/face_swapped_images"
    output_folder = f"results/{folder_number}/final_result"

    tester = InpaintingTester(checkpoint_path, device=device)
    tester.test_on_folder(
        test_folder=input_folder,
        output_folder=output_folder,
        box_size=box_size,
        overlap_ratio=overlap_ratio,
        target_size=target_size
    )
    print(f"Done Testing on folder_number={folder_number}!")

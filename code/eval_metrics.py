import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import ToTensor, Normalize, Compose
import lpips
from scipy.linalg import sqrtm


class EvalMetrics:
    """
    AUTO GPU/CPU evaluator:

    GPU  → Full Inception-V3 FID (accurate, slower)
    CPU  → SPEED-FID (fast: no Inception model)
    """

    def __init__(self, device="cuda"):
        self.device = device

        # LPIPS
        try:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.lpips_available = True
        except Exception:
            self.lpips_available = False
            self.lpips_model = None

        # GPU MODE → FULL INCEPTION V3
        self.use_fast_fid = (device == "cpu")

        if not self.use_fast_fid:
            print("📌 Using FULL Inception-V3 FID (GPU mode)")
            self.inception = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1
            ).to(device)
            self.inception.eval()
            self.inception.fc = nn.Identity()
        else:
            print("⚡ Using SPEED-FID (CPU-optimized, no Inception).")
            self.inception = None

        # Novelty buffer
        self.memory_buffer = None
        self.buffer_size = 1000

        # Normalization transform
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        
    # FEATURE EXTRACTION (GPU → Inception, CPU → simple embedding)
    def get_features(self, images):
        """
        GPU → Extract Inception features
        CPU → Extract simple statistical 2048D feature (SPEED-FID)
        """
        
        # GPU MODE: True FID via Inception-V3
        if not self.use_fast_fid:
            feats = []
            with torch.no_grad():
                for img in images:
                    if not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                    img = img.unsqueeze(0).to(self.device)
                    out = self.inception(img)
                    feats.append(out.cpu().numpy().reshape(-1))
            return np.stack(feats)

        # CPU MODE: SPEED-FID (cheap 2048D stats)
        feats = []
        for img in images:
            if isinstance(img, torch.Tensor):
                arr = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(img, dtype=np.float32)

            # Flatten + resize to 2048D (simple, fast)
            flat = arr.reshape(-1)
            if flat.size > 2048:
                flat = flat[:2048]
            elif flat.size < 2048:
                flat = np.pad(flat, (0, 2048 - flat.size))

            feats.append(flat)

        return np.stack(feats)

    # FID COMPUTATION
    def compute_fid(self, real, gen):
        real = np.asarray(real)
        gen = np.asarray(gen)

        if len(real) == 0 or len(gen) == 0:
            return float('inf')

        mu1, mu2 = real.mean(0), gen.mean(0)
        C1 = np.cov(real, rowvar=False)
        C2 = np.cov(gen, rowvar=False)

        covmean = sqrtm(C1.dot(C2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ((mu1 - mu2) ** 2).sum() + np.trace(C1 + C2 - 2 * covmean)
        return float(fid)

    # LPIPS
    def compute_lpips(self, real_images, gen_images):
        if not self.lpips_available:
            return 0.5

        vals = []
        with torch.no_grad():
            for r, g in zip(real_images, gen_images):
                if not isinstance(r, torch.Tensor):
                    r = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).to(self.device)
                if not isinstance(g, torch.Tensor):
                    g = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(self.device)

                vals.append(self.lpips_model(r, g).item())

        return float(np.mean(vals)) if vals else 0.5

    # DIVERSITY / NOVELTY
    def compute_diversity(self, feats):
        cov = np.cov(feats, rowvar=False) + np.eye(feats.shape[-1]) * 1e-6
        sign, logdet = np.linalg.slogdet(cov)
        return logdet if sign > 0 else -100.0

    def compute_novelty(self, feats):
        if self.memory_buffer is None:
            self.memory_buffer = feats.copy()
            return 0.0

        dists = [np.min(np.linalg.norm(self.memory_buffer - f, axis=1))
                 for f in feats]

        # Update buffer
        self.memory_buffer = np.vstack([self.memory_buffer, feats])
        if len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer = self.memory_buffer[-self.buffer_size:]

        return float(np.mean(dists))

    # FULL EVALUATION ENTRYPOINT
    def evaluate(self, real_imgs, gen_imgs):
        real_feats = self.get_features(real_imgs)
        gen_feats = self.get_features(gen_imgs)

        fid = self.compute_fid(real_feats, gen_feats)
        diversity = self.compute_diversity(gen_feats)
        novelty = self.compute_novelty(gen_feats)
        lpips = self.compute_lpips(real_imgs, gen_imgs) if self.lpips_available else 0.5

        return {
            "FID": fid,
            "diversity": diversity,
            "novelty": novelty,
            "LPIPS": lpips
        }
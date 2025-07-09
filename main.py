import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# モデル定義とハイパーパラメータ
latent_dim = 10
n_classes = 10
img_size = 28

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの初期化と重みロード（weights_only=False を使用）
generator = Generator(latent_dim, n_classes, img_size).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))  # weights_only=False（学習時と同じモデル構造が必要）
generator.eval()

# Streamlit UI
st.title("Conditional GAN Digit Generator")
st.write("好きな数字（0〜9）を選んで、画像を生成してみましょう。")

label = st.selectbox("ラベルを選択してください（数字0〜9）", list(range(10)))
generate_button = st.button("画像を生成")

if generate_button:
    z = torch.randn(1, latent_dim).to(device)
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)
    with torch.no_grad():
        generated_img = generator(z, label_tensor)
    img_np = generated_img.squeeze().cpu().numpy()
    img_np = (img_np + 1) / 2.0  # [-1,1] → [0,1] にスケーリング

    st.write(f"ラベル: {label} に対応する生成画像")
    st.image(img_np, width=224, caption="生成画像", channels="L")

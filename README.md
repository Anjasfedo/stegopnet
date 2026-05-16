# StegoPNet: Deep Image Steganography via Pyramid Pooling

---

## 📜 Academic Attribution

This implementation and research are directly based on the architecture and methodologies proposed in the following academic publication:

> **"StegoPNet: Image Steganography With Generalization Ability Based on Pyramid Pooling Module"** > **Authors:** X. Duan, K. Jia, B. Li, D. Guo, Z. Zhang, and E. Sun
> **Journal:** IEEE Access, 2020
> **DOI:** [10.1109/ACCESS.2020.3033895](https://doi.org/10.1109/ACCESS.2020.3033895)

All architectural foundations, including the integration of the Pyramid Pooling Module (PPM) for multi-scale feature extraction in steganographic tasks, are attributed to the original authors.

---

## 📌 Project Overview

StegoPNet is an end-to-end deep learning framework designed to hide a full-sized RGB secret image within a cover image of identical dimensions ($256 \times 256$). By leveraging a **Pyramid Pooling Module (PPM)**, the network gains a global understanding of the cover image's structure. This allows the model to embed data more intelligently in areas where changes are less perceptible to the human eye and statistical analysis tools.

---

## 🧬 Key Component: Pyramid Pooling Module (PPM)

Unlike standard CNNs that focus on local pixel neighborhoods, the PPM captures features at multiple scales ($32 \times 32$, $16 \times 16$, $8 \times 8$, $4 \times 4$, and $2 \times 2$). This multi-scale approach is crucial for:

* **Global Context Awareness:** Recognizing large, smooth areas versus high-texture regions.
* **Adaptive Embedding:** Prioritizing edges and complex textures to minimize visual artifacts.

---

## 🧮 Mathematical Framework

The total loss of the system is optimized using a weighted Mean Squared Error ($MSE$):

$$Loss = L_{h} + \alpha L_{r}$$

Where:

* $L_{h}$ is the Hiding Loss ($MSE$ between Cover and Stego).
* $L_{r}$ is the Reveal Loss ($MSE$ between Secret and Revealed).
* $\alpha$ is the hyperparameter set to **0.6** to balance the priority between invisibility and reconstruction.

---

## 🚀 Getting Started

### 1. Installation

Ensure you are using **Python 3.6.5** and **PyTorch 1.2.0**.

```bash
pip install torch==1.2.0 torchvision==0.4.0 numpy scikit-image

```

### 2. Dataset Setup

To replicate the research, use the **ImageNet-256** dataset (or any subset of high-quality images). Organize your `data/` directory as follows:

```text
StegoPNet/
└── data/
    ├── train/   # ~45,000 images (Subfolders for classes)
    ├── val/     # ~5,000 images
    └── test/    # ~5,000 images

```

*Note: For the trial run, we utilized the classic **Lena** and **Baboon** pair.*

### 3. Running Training

You can toggle the PPM module within the `train.py` script. To run the full ablation study, you should train both versions:

* **To train with PPM (Proposed):** Ensure `h_net = HidingNet(use_ppm=True)` is set in `train.py`.
* **To train without PPM (Baseline):** Set `use_ppm=False` in the model initialization.

Run the training script:

```bash
python train.py

```

### 4. Monitoring Results

* **Logs:** Training loss (Hiding vs Reveal) is saved to `results/log_train.csv`.
* **Visual Progress:** The script automatically saves triplets (Cover, Stego, Reveal) every 500 iterations in the `results/` folder. Check these to see how well the "camouflage" is evolving.
* **Checkpoints:** Model weights are saved as `.pth` files in the `checkpoints/` folder every epoch.

### 5. Evaluation

Once training is complete, use `evaluate.py` to calculate final metrics and prepare data for steganalysis.

```bash
python evaluate.py

```

This script will:

1. Calculate average **PSNR**, **SSIM**, and **MSE**.
2. Generate **Error Maps** to visualize pixel differences.
3. Save clean and stego image pairs for **StegExpose** analysis.

---

## 🧪 Experimental Results (Trial Run)

The following results were obtained from a trial execution conducted in **Google Colab (Tesla T4 GPU)** using the classic **Lena** image as the cover and the **Baboon** image as the secret payload. The models were trained for **3,000 iterations**.

### 1. Visual Performance & Error Analysis

The comparison below highlights the difference in reconstruction quality and embedding strategy between the Baseline (No PPM) and the proposed StegoPNet (With PPM).

![Visual Performance & Error Analysis](assets/error_map.png)

* **No PPM:** Shows noticeable visual distortion in the Stego image. The Error Map ($\times 10$) reveals a less organized embedding pattern, leading to easier detection.
* **With PPM:** The Stego image remains visually indistinguishable from the original. The Error Map shows that PPM concentrates modifications in textured areas (like the feathers in Lena's hat), significantly improving **imperceptibility**.

### 2. Training Convergence

The training curve demonstrates the stability and efficiency gain provided by the multi-scale pooling architecture.

![Training Convergence](assets/training_curve.png)

* **PPM (Orange Line):** Exhibits a smoother and faster descent. It achieves a significantly lower loss, proving that the PPM helps the network solve the "hiding" and "revealing" tasks more effectively.
* **No PPM (Blue Line):** Displays higher volatility and plateaus at a higher loss value. The sharp spikes suggest that without multi-scale features, the network struggles to find a stable way to hide the high-entropy Baboon data.

---

*Developed for academic research purposes. Last updated May 2026.*

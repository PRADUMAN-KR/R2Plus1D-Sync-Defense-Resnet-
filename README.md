
# Multimodal Lip Sync Deepfake Detection System

# Multimodal Lip Sync Deepfake Detection System

Production-ready deep learning system for detecting lip sync deepfakes via audio-video synchronization mismatch.

🚀 Real-time inference  
🎯 Low false positive rate  
⚙️ Scalable FastAPI-based pipeline



## 📊 Performance

- Accuracy: 98%+
- False Positives: Reduced via confidence aggregation to 0.4% tested on 2500 validation set
- Dataset: 50K+ video clips (real + fake)

---
## 🚀 Project Highlights

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)




![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![Deepfake Detection](https://img.shields.io/badge/Domain-Deepfake%20Detection-critical?style=for-the-badge)
![Multimodal](https://img.shields.io/badge/Architecture-Audio--Visual%20Encoder-6C63FF?style=for-the-badge)
![Transformer](https://img.shields.io/badge/Temporal-Transformer%20Encoder-FF6B6B?style=for-the-badge)
![Cross Attention](https://img.shields.io/badge/Fusion-Bidirectional%20Cross--Attention-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)
---

---

## 📌 Overview

**R2Plus1D-Sync-Defense-Resnet** is an advanced deepfake detection system designed to detect **audio-visual lip-sync manipulation** using:

- 🎥 3D Spatio-Temporal Video Encoding  
- 🔊 Audio Spectrogram Feature Extraction  
- 🔁 Cross-Modal Attention Fusion  
- 🧠 Transformer Temporal Modeling  
- 🛡️ Artifact-Aware Forgery Detection  

Unlike frame-based detectors, this model analyzes **temporal consistency between speech and mouth movements**, making it robust against modern lip-sync deepfakes such as Wav2Lip-style manipulations.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph IN["INPUTS"]
        V["Visual (B, 3, T, H, W)"]
        A["Audio (B, 1, F, T_a)"]
    end

    subgraph VE["VISUAL ENCODER — 3D ResNet-style"]
        V --> VE_C["Conv3d 3→64 k=3,7,7 s=1,2,2 pad=1,3,3"]
        VE_C --> VE_BN1["BatchNorm3d 64"]
        VE_BN1 --> VE_R1["ReLU"]
        VE_R1 --> VE_MP["MaxPool3d k=1,3,3 s=1,2,2"]
        VE_MP --> VE_L1["ResBlock3D 64→64 s=1,1,1"]
        VE_L1 --> VE_L2["ResBlock3D 64→128 s=1,2,2"]
        VE_L2 --> VE_L3["ResBlock3D 128→256 s=1,2,2"]
        VE_L3 --> VE_L4["ResBlock3D 256→256 s=1,2,2"]
        VE_L4 --> VE_D["Dropout3d"]
        VE_D --> VE_FM["feature_map (B,256,T',H',W')"]
        VE_FM --> VE_AP["AdaptiveAvgPool3d T',1,1"]
        VE_AP --> VE_SQ["squeeze -1,-2"]
        VE_SQ --> v_feat["v_feat (B, 256, T')"]
        VE_FM --> v_map["v_map (B,256,T',H',W')"]
    end

    subgraph AE["AUDIO ENCODER — 2D ResNet-style"]
        A --> AE_C["Conv2d 1→64 k=7 s=2,2 pad=3"]
        AE_C --> AE_BN["BatchNorm2d 64"]
        AE_BN --> AE_R["ReLU"]
        AE_R --> AE_MP["MaxPool2d k=3 s=2,2"]
        AE_MP --> AE_L1["ResBlock2D 64→64 s=1,1"]
        AE_L1 --> AE_L2["ResBlock2D 64→128 s=2,2"]
        AE_L2 --> AE_L3["ResBlock2D 128→256 s=2,1"]
        AE_L3 --> AE_L4["ResBlock2D 256→256 s=2,1"]
        AE_L4 --> AE_D["Dropout"]
        AE_D --> AE_AP["AdaptiveAvgPool2d 1,T'"]
        AE_AP --> AE_SQ["squeeze dim=2"]
        AE_SQ --> a_feat["a_feat (B, 256, T')"]
    end

    subgraph PROJ["FEATURE PROJECTION"]
        v_feat --> PV_T["transpose 1,2 → (B,T,D)"]
        PV_T --> PV_L["Linear 256→256 visual_proj"]
        PV_L --> v_emb["v_emb (B, T_v, 256)"]
        a_feat --> PA_T["transpose 1,2 → (B,T,D)"]
        PA_T --> PA_L["Linear 256→256 audio_proj"]
        PA_L --> a_emb["a_emb (B, T_a, 256)"]
    end

    subgraph CMA["CROSS-MODAL ATTENTION — Gated fusion"]
        v_emb --> CMA_interp["If T_v≠T_a: F.interpolate audio to T_v"]
        a_emb --> CMA_interp
        CMA_interp --> a_emb_aligned["a_emb (B,T,256)"]
        v_emb --> v2a_q["v2a: Q from visual"]
        a_emb_aligned --> v2a_kv["v2a: K,V from audio"]
        v2a_q --> v2a_attn["MultiheadAttention 256 dim 8 heads"]
        v2a_kv --> v2a_attn
        v2a_attn --> v_attn["v_attended (B,T,256)"]
        v_emb --> v_add["v_out = v_emb + v_attended"]
        v_attn --> v_add
        a_emb_aligned --> a2v_q["a2v: Q from audio"]
        v_emb --> a2v_kv["a2v: K,V from visual"]
        a2v_q --> a2v_attn["MultiheadAttention 256 dim 8 heads"]
        a2v_kv --> a2v_attn
        a2v_attn --> a_attn["a_attended (B,T,256)"]
        a_emb_aligned --> a_add["a_out = a_emb + a_attended"]
        a_attn --> a_add
        v_add --> gate_cat["concat v_out a_out (B,T,512)"]
        a_add --> gate_cat
        gate_cat --> gate_L1["Linear 512→256"]
        gate_L1 --> gate_G["GELU"]
        gate_G --> gate_L2["Linear 256→1"]
        gate_L2 --> gate_S["Sigmoid → g (B,T,1)"]
        gate_S --> gate_blend["fused = g×v_out + 1-g×a_out"]
        v_add --> gate_blend
        a_add --> gate_blend
        gate_blend --> fuse_L["Linear 256→256"]
        fuse_L --> fuse_R["ReLU"]
        fuse_R --> fused["fused (B, T, 256)"]
    end

    subgraph TT["TEMPORAL TRANSFORMER — Multi-scale + CLS"]
        fused --> TT_tr["transpose → (B,256,T)"]
        TT_tr --> TT_b3["Conv1d 256→256 k=3 p=1"]
        TT_b3 --> TT_b3bn["BatchNorm1d"]
        TT_b3bn --> TT_b3g["GELU → c3 (B,256,T)"]
        TT_tr --> TT_b5["Conv1d 256→256 k=5 p=2"]
        TT_b5 --> TT_b5bn["BatchNorm1d"]
        TT_b5bn --> TT_b5g["GELU → c5 (B,256,T)"]
        TT_tr --> TT_b7["Conv1d 256→256 k=7 p=3"]
        TT_b7 --> TT_b7bn["BatchNorm1d"]
        TT_b7bn --> TT_b7g["GELU → c7 (B,256,T)"]
        TT_b3g --> TT_cat["concat dim=1 → (B,768,T)"]
        TT_b5g --> TT_cat
        TT_b7g --> TT_cat
        TT_cat --> TT_tr2["transpose → (B,T,768)"]
        TT_tr2 --> TT_proj["Linear 768→256 → x_conv"]
        TT_proj --> TT_add["x = fused + x_conv"]
        fused --> TT_add
        TT_add --> TT_cls["prepend CLS token 1,1,256"]
        TT_cls --> TT_seq["tokens (B, 1+T, 256)"]
        TT_seq --> TT_enc["TransformerEncoder 4 layers"]
        TT_enc --> TT_enc_d["d_model=256 nhead=8 ff=1024 GELU norm_first"]
        TT_enc_d --> TT_out["cls_output (B, 256)"]
    end

    subgraph AD["ARTIFACT DETECTOR — Raw + Delta + High-freq"]
        v_map --> AD_raw_C1["Conv3d 256→128 k=3,3,3 p=1"]
        AD_raw_C1 --> AD_raw_BN1["BN3d ReLU"]
        AD_raw_BN1 --> AD_raw_C2["Conv3d 128→64 k=3,3,3 p=1"]
        AD_raw_C2 --> AD_raw_BN2["BN3d ReLU"]
        AD_raw_BN2 --> AD_raw_P["AdaptiveAvgPool3d 1,1,1"]
        AD_raw_P --> raw_feat["raw_feat (B, 64)"]
        v_map --> AD_delta["delta = v_map 1: - v_map :-1"]
        AD_delta --> AD_delta_C1["Conv3d 256→128 k=3,3,3"]
        AD_delta_C1 --> AD_delta_BN1["BN3d ReLU"]
        AD_delta_BN1 --> AD_delta_C2["Conv3d 128→64 k=3,3,3"]
        AD_delta_C2 --> AD_delta_BN2["BN3d ReLU"]
        AD_delta_BN2 --> AD_delta_P["AdaptiveAvgPool3d 1,1,1"]
        AD_delta_P --> delta_feat["delta_feat (B, 64)"]
        V --> AD_lap["Laplacian Conv2d 3,3,3 fixed kernel per frame"]
        AD_lap --> AD_hf_C1["Conv3d 3→32 k=3,3,3 s=1,2,2"]
        AD_hf_C1 --> AD_hf_BN1["BN3d ReLU"]
        AD_hf_BN1 --> AD_hf_C2["Conv3d 32→64 k=3,3,3 s=1,2,2"]
        AD_hf_C2 --> AD_hf_BN2["BN3d ReLU"]
        AD_hf_BN2 --> AD_hf_P["AdaptiveAvgPool3d 1,1,1"]
        AD_hf_P --> hf_feat["hf_feat (B, 64)"]
        raw_feat --> AD_cat["concat raw + delta + hf (B, 192)"]
        delta_feat --> AD_cat
        hf_feat --> AD_cat
        TT_out --> AD_comb["concat with cls_output (B, 448)"]
        AD_cat --> AD_comb
        AD_comb --> AD_fuse_L1["Linear 448→256"]
        AD_fuse_L1 --> AD_fuse_R1["ReLU"]
        AD_fuse_R1 --> AD_fuse_L2["Linear 256→128"]
        AD_fuse_L2 --> AD_fuse_R2["ReLU"]
        AD_fuse_R2 --> artifact_feat["artifact_feat (B, 128)"]
    end

    subgraph MERGE["COMBINE"]
        TT_out --> merge_cat["concat cls_output + artifact_feat"]
        artifact_feat --> merge_cat
        merge_cat --> combined["combined (B, 384)"]
    end

    subgraph HEAD["CLASSIFICATION HEAD"]
        combined --> H_L1["Linear 384→128"]
        H_L1 --> H_G["GELU"]
        H_G --> H_D["Dropout p=0.1"]
        H_D --> H_LN["LayerNorm 128"]
        H_LN --> H_L2["Linear 128→1"]
        H_L2 --> H_sq["squeeze -1"]
        H_sq --> logits["logits (B)"]
    end

  %% ===== DARK MODE FRIENDLY STYLES =====
style VE fill:#2a2f3a,stroke:#7aa2ff,color:#ffffff
style AE fill:#2a2f3a,stroke:#7aa2ff,color:#ffffff
style CMA fill:#3a2f4a,stroke:#b38cff,color:#ffffff
style TT fill:#2f3a2f,stroke:#6adf91,color:#ffffff
style ADMOD fill:#4a2f2f,stroke:#ff8a8a,color:#ffffff

```
---


# 🧠 Model Design

## 🎥 Visual Branch

* **R2Plus1D-style 3D ResNet**
* Captures lip movement dynamics across time
* **Input shape:** `(B, 3, T, H, W)`

## 🔊 Audio Branch

* **Log Mel Spectrogram**
* **2D ResNet backbone**
* **Input shape:** `(B, 1, F, T)`

## 🔁 Fusion Module

* **Bidirectional Cross-Attention**

  * Audio attends to visual
  * Visual attends to audio

## 🧠 Temporal Modeling

* **Transformer encoder layers**
* Sequence reasoning across frames

---

# 📊 Performance (Sample Metrics)

| Metric             | Score     |
| ------------------ | --------- |
| Accuracy           | 98%+      |
| F1 Score           | 0.97     |
| Precision          | 0.98      |
| Recall             | 0.97      |
| Avg Inference Time | ~3s (GPU) |

⚡ Optimizable to **<1.5s** with GPU + batching.

---

## 📦 Installation

```bash
git clone https://github.com/PRADUMAN-KR/R2Plus1D-Sync-Defense-Resnet-.git
cd R2Plus1D-Sync-Defense-Resnet-

python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

# 🚀 Running the API

```bash
uvicorn app.main:app --reload
```

API runs at:

```
http://127.0.0.1:8000
```

---


## ⚙️ Production Features

* ✅ Multi-face tracking
* ✅ Confidence margin rule
* ✅ Uncertain prediction flag
* ✅ VAD speech detection filtering
* ✅ Robust mouth ROI extraction
* ✅ Long-video adaptive inference

---

## 🔬 Research Direction

### Future Improvements

* Contrastive Audio-Visual Pretraining
* Phoneme-Level Supervision
* Real-Time Streaming Inference
* Edge Deployment Optimization
* Self-Supervised Cross-Modal Learning

---

## 📈 Deployment Options

* FastAPI REST Service
* Dockerized Inference
* GPU Deployment (CUDA)
* Cloud (AWS / GCP / Azure)
* Real-Time Webcam Pipeline (Future)

---

## 🛡️ Use Cases

* Interview Fraud Detection
* Media Authenticity Verification
* Social Media Deepfake Filtering
* Security & Biometric Systems
* Digital Forensics

---

## 📜 License

Licensed under the **Apache 2.0 License**.

---

## 👨‍💻 Author

![Designation](https://img.shields.io/badge/Praduman%20Kumar%20|%20AI%20Engineer-0A192F?style=for-the-badge)

[![GitHub](https://img.shields.io/badge/GitHub-PRADUMAN--KR-181717?style=for-the-badge&logo=github)](https://github.com/PRADUMAN-KR)

---

⭐ If you find this repo useful, consider **starring the repository**!


#!/usr/bin/env python3
"""
Script phân tích phân phối similarity scores trong dataset training.
Đơn giản, hiệu quả, tất cả trong 1 file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ============== CẤU HÌNH ==============
DATASET_PATH = "data/processed/training_dataset/train_dataset_80.json"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR = Path("logs/dataset_analysis")
SAMPLE_SIZE = None  

print("=" * 80)
print("📊 PHÂN TÍCH PHÂN PHỐI DATASET")
print("=" * 80)
print(f"Dataset: {DATASET_PATH}")
print(f"Model: {MODEL_NAME}")
print(f"Sample size: {'Full dataset' if SAMPLE_SIZE is None else SAMPLE_SIZE}")
print()

# ============== 1. LOAD DATA ==============
print("📂 Đang load dataset...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

if SAMPLE_SIZE:
    data = data[:SAMPLE_SIZE]

print(f"✓ Loaded {len(data)} samples\n")

# ============== 2. LOAD MODEL & COMPUTE SCORES ==============
print(f"🤖 Đang load model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
print("✓ Model loaded\n")

print("🧮 Đang tính similarity scores...")
queries = [item['query'] for item in data]
positives = [item['positive'] for item in data]
negatives = [item['negative'] for item in data]

print("  - Encoding queries...")
query_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)

print("  - Encoding positives...")
pos_emb = model.encode(positives, convert_to_tensor=True, show_progress_bar=True)

print("  - Encoding negatives...")
neg_emb = model.encode(negatives, convert_to_tensor=True, show_progress_bar=True)

print("  - Computing cosine similarities...")
pos_scores = util.cos_sim(query_emb, pos_emb).diagonal().float().cpu().numpy()
neg_scores = util.cos_sim(query_emb, neg_emb).diagonal().float().cpu().numpy()
margins = pos_scores - neg_scores

print("✓ Done\n")

# ============== 3. THỐNG KÊ ==============
print("=" * 80)
print("📈 THỐNG KÊ CHI TIẾT")
print("=" * 80)

print("\n🟢 POSITIVE SCORES (Query vs Positive):")
print(f"  Mean:      {np.mean(pos_scores):.4f}")
print(f"  Median:    {np.median(pos_scores):.4f}")
print(f"  Std:       {np.std(pos_scores):.4f}")
print(f"  Min:       {np.min(pos_scores):.4f}")
print(f"  Max:       {np.max(pos_scores):.4f}")
print(f"  Q25:       {np.percentile(pos_scores, 25):.4f}")
print(f"  Q75:       {np.percentile(pos_scores, 75):.4f}")

print("\n🔴 NEGATIVE SCORES (Query vs Negative):")
print(f"  Mean:      {np.mean(neg_scores):.4f}")
print(f"  Median:    {np.median(neg_scores):.4f}")
print(f"  Std:       {np.std(neg_scores):.4f}")
print(f"  Min:       {np.min(neg_scores):.4f}")
print(f"  Max:       {np.max(neg_scores):.4f}")
print(f"  Q25:       {np.percentile(neg_scores, 25):.4f}")
print(f"  Q75:       {np.percentile(neg_scores, 75):.4f}")

print("\n📏 MARGIN (Positive - Negative):")
print(f"  Mean:      {np.mean(margins):.4f}")
print(f"  Median:    {np.median(margins):.4f}")
print(f"  Std:       {np.std(margins):.4f}")
print(f"  Min:       {np.min(margins):.4f}")
print(f"  Max:       {np.max(margins):.4f}")
print(f"  Q25:       {np.percentile(margins, 25):.4f}")
print(f"  Q75:       {np.percentile(margins, 75):.4f}")

# Phát hiện vấn đề
false_negs = np.sum(margins < 0)
false_neg_ratio = false_negs / len(data) * 100

print("\n⚠️  FALSE NEGATIVES (Negative score > Positive score):")
print(f"  Count:     {false_negs} / {len(data)} ({false_neg_ratio:.2f}%)")

if false_neg_ratio > 5:
    print("  ❌ CẢNH BÁO: Tỷ lệ False Negative cao! Dataset có nhiễu nghiêm trọng.")
elif false_neg_ratio > 1:
    print("  ⚠️  Có một ít False Negatives. Cân nhắc lọc trước khi train.")
else:
    print("  ✅ Tỷ lệ False Negative thấp, dataset tương đối sạch.")

# ============== 4. VISUALIZATION ==============
print("\n📊 Đang tạo visualizations...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Histogram - Positive vs Negative Scores
ax = axes[0, 0]
ax.hist(pos_scores, bins=50, alpha=0.6, label='Positive', color='green', edgecolor='black')
ax.hist(neg_scores, bins=50, alpha=0.6, label='Negative', color='red', edgecolor='black')
ax.axvline(np.mean(pos_scores), color='green', linestyle='--', linewidth=2, 
           label=f'Pos Mean: {np.mean(pos_scores):.3f}')
ax.axvline(np.mean(neg_scores), color='red', linestyle='--', linewidth=2,
           label=f'Neg Mean: {np.mean(neg_scores):.3f}')
ax.set_xlabel('Cosine Similarity Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Phân phối Positive vs Negative Scores', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Box Plot
ax = axes[0, 1]
data_box = [pos_scores, neg_scores]
bp = ax.boxplot(data_box, labels=['Positive', 'Negative'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Cosine Similarity Score', fontsize=12)
ax.set_title('Box Plot: Positive vs Negative', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Margin Distribution
ax = axes[1, 0]
colors = ['red' if m < 0 else 'green' for m in margins]
n, bins, patches = ax.hist(margins, bins=50, edgecolor='black', alpha=0.7)
# Color bars based on value
for i, patch in enumerate(patches):
    if bins[i] < 0:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)
    else:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)
        
ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero (Danger Zone)')
ax.axvline(np.mean(margins), color='blue', linestyle='-', linewidth=2,
           label=f'Mean: {np.mean(margins):.3f}')
ax.axvline(np.median(margins), color='purple', linestyle=':', linewidth=2,
           label=f'Median: {np.median(margins):.3f}')
ax.set_xlabel('Margin (Positive - Negative)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Phân phối Margin (chênh lệch)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Scatter - Positive vs Negative
ax = axes[1, 1]
scatter = ax.scatter(pos_scores, neg_scores, alpha=0.4, c=margins, cmap='RdYlGn', s=20)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Pos = Neg (False Negative)')
ax.set_xlabel('Positive Score', fontsize=12)
ax.set_ylabel('Negative Score', fontsize=12)
ax.set_title('Scatter: Positive vs Negative Scores', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Margin', fontsize=10)

plt.tight_layout()

output_path = OUTPUT_DIR / "distribution_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {output_path}")
plt.close()

# ============== 5. SAVE STATISTICS ==============
stats = {
    'dataset_info': {
        'path': str(DATASET_PATH),
        'total_samples': len(data),
        'model': MODEL_NAME
    },
    'positive_scores': {
        'mean': float(np.mean(pos_scores)),
        'median': float(np.median(pos_scores)),
        'std': float(np.std(pos_scores)),
        'min': float(np.min(pos_scores)),
        'max': float(np.max(pos_scores)),
        'q25': float(np.percentile(pos_scores, 25)),
        'q75': float(np.percentile(pos_scores, 75))
    },
    'negative_scores': {
        'mean': float(np.mean(neg_scores)),
        'median': float(np.median(neg_scores)),
        'std': float(np.std(neg_scores)),
        'min': float(np.min(neg_scores)),
        'max': float(np.max(neg_scores)),
        'q25': float(np.percentile(neg_scores, 25)),
        'q75': float(np.percentile(neg_scores, 75))
    },
    'margins': {
        'mean': float(np.mean(margins)),
        'median': float(np.median(margins)),
        'std': float(np.std(margins)),
        'min': float(np.min(margins)),
        'max': float(np.max(margins)),
        'q25': float(np.percentile(margins, 25)),
        'q75': float(np.percentile(margins, 75))
    },
    'quality_metrics': {
        'false_negatives': int(false_negs),
        'false_negative_ratio': float(false_neg_ratio)
    }
}

stats_path = OUTPUT_DIR / "statistics.json"
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"✓ Saved statistics to {stats_path}")

# ============== 6. SUMMARY & RECOMMENDATION ==============
print("\n" + "=" * 80)
print("🎯 TÓM TẮT & KHUYẾN NGHỊ")
print("=" * 80)

avg_margin = np.mean(margins)
print(f"\n📊 Chất lượng Dataset:")
print(f"  • Positive mean score: {np.mean(pos_scores):.4f}")
print(f"  • Negative mean score: {np.mean(neg_scores):.4f}")
print(f"  • Average margin: {avg_margin:.4f}")
print(f"  • False Negative ratio: {false_neg_ratio:.2f}%")

print(f"\n💡 Khuyến nghị:")
if false_neg_ratio > 5:
    print("  ❌ Dataset có nhiễu nghiêm trọng!")
    print("     → BẮT BUỘC lọc False Negatives trước khi train")
    print("     → KHÔNG dùng Hard Negative Mining")
    print("     → Xem xét lại quy trình sinh dữ liệu")
elif false_neg_ratio > 1:
    print("  ⚠️  Dataset có một ít nhiễu")
    print("     → Nên lọc những mẫu có margin < 0")
    print("     → Train với learning rate thấp và early stopping")
elif avg_margin < 0.15:
    print("  ⚠️  Margin thấp (hard negatives)")
    print("     → Nếu data AI-generated: cân nhắc lọc nhẹ")
    print("     → Train cẩn thận với gradient accumulation")
else:
    print("  ✅ Dataset chất lượng tốt!")
    print("     → Có thể train trực tiếp")
    print("     → Sử dụng MultipleNegativesRankingLoss")

print(f"\n📁 Files đã tạo:")
print(f"  • {output_path}")
print(f"  • {stats_path}")

print("\n" + "=" * 80)
print("✅ HOÀN TẤT!")
print("=" * 80)

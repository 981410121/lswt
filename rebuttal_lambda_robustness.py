import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 填入真实实验数据 =================
lambda_vals =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# --- 实验 1: LLaVA-1.5-7B ---
llava_50 =[1600, 1599, 1603, 1608, 1613, 1608, 1598, 1608, 1605, 1604, 1607]
llava_75 =[1579, 1563, 1570, 1575, 1592, 1607, 1588, 1578, 1568, 1575, 1571]
llava_90 =[1425, 1438, 1441, 1421, 1434, 1456, 1451, 1451, 1456, 1452, 1437]

# --- 实验 2: Qwen2.5-VL-3B (840x840) ---
qwen840_50 =[1851, 1850, 1861, 1873, 1910, 1912, 1900, 1876, 1902, 1901, 1909]
qwen840_66 =[1804, 1800, 1810, 1807, 1813, 1829, 1801, 1822, 1812, 1815, 1813]
qwen840_75 =[1768, 1772, 1770, 1760, 1776, 1785, 1775, 1761, 1771, 1765, 1761]

# --- 实验 3: Qwen2.5-VL-3B (1120x1120) ---
qwen1120_50 =[1890, 1888, 1903, 1902, 1915, 1913, 1885, 1902, 1897, 1894, 1900]
qwen1120_66 =[1849, 1830, 1851, 1845, 1862, 1885, 1848, 1848, 1835, 1845, 1841]
qwen1120_75 =[1795, 1808, 1804, 1781, 1804, 1843, 1812, 1831, 1843, 1833, 1817]
# =========================================================

# 2. 顶刊级样式设置
sns.set_theme(style="ticks") 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams["font.size"] = 11

# 高对比度配色 (蓝, 红, 绿)
colors =["#1F77B4", "#D62728", "#2CA02C"] 

# 3. 创建画布，略微增加高度，给图例留出充足空间
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

def plot_subplot(ax, data_level1, data_level2, data_level3, title):
    # 【修复1】大幅缩小 markersize 到 5，白边 markeredgewidth 减小到 1.0，整体显得更秀气
    l1, = ax.plot(lambda_vals, data_level1, marker='o', linewidth=2.0, markersize=5, 
                  color=colors[0], markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    l2, = ax.plot(lambda_vals, data_level2, marker='s', linewidth=2.0, markersize=5, 
                  color=colors[1], markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    l3, = ax.plot(lambda_vals, data_level3, marker='^', linewidth=2.0, markersize=5, 
                  color=colors[2], markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(r"Weighting Coefficient ($\lambda$)", fontsize=11, labelpad=6)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray', zorder=0)
    
    all_data = data_level1 + data_level2 + data_level3
    y_min, y_max = min(all_data), max(all_data)
    padding = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - padding, y_max + padding)
    
    return [l1, l2, l3]

# 4. 绘制三个子图
lines = plot_subplot(axes[0], llava_50, llava_75, llava_90, "LLaVA-1.5-7B")
plot_subplot(axes[1], qwen840_50, qwen840_66, qwen840_75, "Qwen2.5-VL (840×840)")
plot_subplot(axes[2], qwen1120_50, qwen1120_66, qwen1120_75, "Qwen2.5-VL (1120×1120)")

axes[0].set_ylabel("MME Score", fontsize=12, fontweight='bold', labelpad=8)
sns.despine()

# 5. 【修复2】强制绑定图例到中间的子图 (axes[1])，保证图例绝对不会被裁剪！
legend_labels =[
    '50% Drop', 
    '75% Drop (LLaVA) / 66% (Qwen)', 
    '90% Drop (LLaVA) / 75% (Qwen)'
]
axes[1].legend(lines, legend_labels, 
               loc='lower center', bbox_to_anchor=(0.5, 1.15), 
               ncol=3, frameon=False, fontsize=11, columnspacing=1.5)

# 6. 调整布局，增加头部边距防止图例重叠
plt.subplots_adjust(top=0.75, bottom=0.15, wspace=0.25)

# 保存文件（bbox_inches="tight" 会自动包含 axes[1] 上的图例）
file_name = "rebuttal_lambda_robustness_final.pdf"
plt.savefig(file_name, format="pdf", bbox_inches="tight")

print("\n" + "="*50)
print(f"✅ 图例已修复！数据点已缩小！")
print(f"📂 真正的 PDF 文件请务必在这里查看（不要看编辑器的临时预览）：\n{os.path.abspath(file_name)}")
print("="*50 + "\n")

plt.show()
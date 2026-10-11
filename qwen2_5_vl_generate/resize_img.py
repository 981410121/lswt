import os
from PIL import Image

def resize_image(input_path, output_path, target_width=1512, target_height=2044, method="stretch"):
    """
    调整图片分辨率
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :param method: "stretch" (强制拉伸) 或 "pad" (保持比例填充)
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 转换颜色模式为RGB（防止PNG透明通道导致保存JPG报错）
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            original_size = img.size
            print(f"原始尺寸: {original_size}")

            if method == "stretch":
                # === 方案1: 强制拉伸 (可能变形) ===
                # Image.Resampling.LANCZOS 是高质量重采样算法
                new_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            elif method == "pad":
                # === 方案2: 保持比例，填充背景 (不会变形) ===
                # 1. 计算缩放比例
                ratio = min(target_width / original_size[0], target_height / original_size[1])
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                
                # 2. 缩放原图
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 3. 创建黑色背景的新图
                new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0)) # (0,0,0)是黑色
                
                # 4. 将缩放后的图粘贴到中心
                paste_x = (target_width - new_size[0]) // 2
                paste_y = (target_height - new_size[1]) // 2
                new_img.paste(img_resized, (paste_x, paste_y))

            else:
                print("未知的处理方法")
                return

            # 保存图片
            new_img.save(output_path, quality=95) # quality=95 保证JPG质量
            print(f"成功! 图片已保存至: {output_path} (尺寸: {target_width}x{target_height})")

    except Exception as e:
        print(f"处理出错: {e}")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 在这里修改你的图片路径
    input_file = "/root/final_paper/插图.jpg"      # 你的原图名字
    output_file = "/root/final_paper/qwen2_5_vl_generate/resized.jpg"   # 你想保存的名字

    # 方式 1: 强制拉伸 (图片会填满 1512*2044，但可能变胖或变瘦)
    resize_image(input_file, output_file, 1024, 1024, method="stretch")

    # 方式 2: 保持比例 (图片不 deformed，周围会有黑边)
    # resize_image(input_file, output_file, 1512, 2044, method="pad")
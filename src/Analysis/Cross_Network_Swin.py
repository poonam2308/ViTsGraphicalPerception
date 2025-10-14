from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve()
IMAGES_DIR = HERE.parents[1] / "Images"

image_paths = [
    IMAGES_DIR / "swin_position_common_scale_cn.png",
    IMAGES_DIR / "swin_length_cn.png",
    IMAGES_DIR / "swin_area_cn.png",
    IMAGES_DIR / "swin_shading_cn.png",
]

# Optional: helpful check
missing = [p for p in image_paths if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing image(s): {', '.join(str(p) for p in missing)}")

images = [Image.open(p) for p in image_paths]


base_width = images[0].width
resized_images = [
    img if img.width == base_width else img.resize((base_width, int(img.height * base_width / img.width)))
    for img in images
]
total_height = sum(img.height for img in resized_images)
combined_image = Image.new("RGB", (base_width, total_height), color=(255, 255, 255))

y_offset = 0
for img in resized_images:
    combined_image.paste(img, (0, y_offset))
    y_offset += img.height

# Save output
combined_image.save(IMAGES_DIR/ "Swin_combined_tasks.png")
# combined_image.save("analysis_pdfs/analysis_pdfs_with_stimulus1/Swin_combined_tasks.pdf")  # Optional: also as PDF

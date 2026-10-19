"""Image preprocessing shared by HAM10000 (training) and DDI (evaluation)."""
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Padding colour = ImageNet mean, so padded borders normalise to roughly zero.
PAD_FILL = tuple(round(255 * m) for m in IMAGENET_MEAN)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB. RGBA is composited over PAD_FILL, which equals dropping
    the alpha channel when the image is fully opaque."""
    if img.mode == "RGBA":
        canvas = Image.new("RGB", img.size, PAD_FILL)
        canvas.paste(img, mask=img.getchannel("A"))
        return canvas
    return img.convert("RGB")


def pad_to_square(img: Image.Image, fill=PAD_FILL) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def prepare_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """RGB -> pad to square (no cropping, aspect ratio kept) -> resize."""
    squared = pad_to_square(to_rgb(img))
    return squared.resize((size, size), Image.Resampling.BICUBIC)
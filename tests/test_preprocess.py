from PIL import Image

from calibration_bias.preprocess import IMAGE_SIZE, PAD_FILL, pad_to_square, prepare_image, to_rgb

RED = (255, 0, 0)


def test_output_is_square_rgb_for_extreme_aspects():
    for w, h in [(100, 300), (300, 100), (224, 224), (79, 155)]:
        out = prepare_image(Image.new("RGB", (w, h), (10, 20, 30)))
        assert out.size == (IMAGE_SIZE, IMAGE_SIZE)
        assert out.mode == "RGB"


def test_padding_keeps_aspect_ratio_and_does_not_crop():
    sq = pad_to_square(Image.new("RGB", (100, 300), RED))
    assert sq.size == (300, 300)
    assert sq.getpixel((0, 150)) == PAD_FILL      # left border is padding
    assert sq.getpixel((99, 150)) == PAD_FILL     # last padding column
    assert sq.getpixel((100, 0)) == RED           # content reaches the top edge
    assert sq.getpixel((199, 299)) == RED         # ...and the bottom edge
    assert sq.getpixel((200, 299)) == PAD_FILL    # right border is padding


def test_rgba_opaque_equals_plain_rgb():
    img = Image.new("RGBA", (50, 50), (200, 100, 50, 255))
    assert to_rgb(img).getpixel((10, 10)) == (200, 100, 50)


def test_rgba_transparent_pixels_become_pad_colour():
    img = Image.new("RGBA", (50, 50), (200, 100, 50, 0))
    assert to_rgb(img).getpixel((10, 10)) == PAD_FILL


def test_other_modes_convert_to_rgb():
    for mode in ("L", "P", "RGB"):
        assert to_rgb(Image.new(mode, (10, 10))).mode == "RGB"
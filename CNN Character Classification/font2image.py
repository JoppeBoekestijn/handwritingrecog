import os
# Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw, ImageFilter


# Load the font and set the font size to 42
font = ImageFont.truetype('habbakuk/Habbakuk.ttf', 42)

# Character mapping for each of the 27 tokens
char_map = {'Alef': ')',
            'Ayin': '(',
            'Bet': 'b',
            'Dalet': 'd',
            'Gimel': 'g',
            'He': 'x',
            'Het': 'h',
            'Kaf': 'k',
            'Kaf-final': '\\',
            'Lamed': 'l',
            'Mem': '{',
            'Mem-medial': 'm',
            'Nun-final': '}',
            'Nun-medial': 'n',
            'Pe': 'p',
            'Pe-final': 'v',
            'Qof': 'q',
            'Resh': 'r',
            'Samekh': 's',
            'Shin': '$',
            'Taw': 't',
            'Tet': '+',
            'Tsadi-final': 'j',
            'Tsadi-medial': 'c',
            'Waw': 'w',
            'Yod': 'y',
            'Zayin': 'z'}


# Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')

    # Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(img)

    # Get size of the font and draw the token in the center of the blank image
    w, h = font.getsize(char_map[label])
    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2),
              char_map[label], 0, font)
    return img


def create_and_save_images():
    # Create directories and store images
    os.mkdir('figures/train')
    os.mkdir('figures/test')
    for key, values in char_map.items():
        os.mkdir('figures/train/' + key)
        os.mkdir('figures/test/' + key)
        img = create_image(key, (56, 56))
        img_blur = img.filter(ImageFilter.BLUR)
        img_detail = img.filter(ImageFilter.DETAIL)
        img_edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
        img_edge_enhance_more = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img_sharpen = img.filter(ImageFilter.SHARPEN)
        img_smooth = img.filter(ImageFilter.SMOOTH)
        img_smooth_more = img.filter(ImageFilter.SMOOTH_MORE)
        img_gaussian_blur_half = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img_gaussian_blur_1 = img.filter(ImageFilter.GaussianBlur(radius=1))
        img_gaussian_blur_1_half = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img_gaussian_blur_2 = img.filter(ImageFilter.GaussianBlur(radius=2))
        img_modefilter_1 = img.filter(ImageFilter.ModeFilter(size=1))
        img_modefilter_3 = img.filter(ImageFilter.ModeFilter(size=3))
        img_modefilter_5 = img.filter(ImageFilter.ModeFilter(size=5))
        img_minfilter_1 = img.filter(ImageFilter.MinFilter(1))
        img_minfilter_3 = img.filter(ImageFilter.MinFilter(3))
        img_minfilter_5 = img.filter(ImageFilter.MinFilter(5))
        img_unsharpmask = img.filter(ImageFilter.UnsharpMask(radius=1,
                                                             percent=150,
                                                             threshold=3))

        img.save('figures/test/' + key + '/standard.png')
        img.save('figures/train/' + key + '/standard.png')
        img_blur.save('figures/train/' + key + '/img_blur.png')
        img_detail.save('figures/train/' + key + '/img_detail.png')
        img_edge_enhance.save('figures/train/' + key + '/img_edge_enhance.png')
        img_edge_enhance_more.save('figures/train/' + key + '/img_edge_enhance_more.png')
        img_sharpen.save('figures/train/' + key + '/img_sharpen.png')
        img_smooth.save('figures/train/' + key + '/img_smooth.png')
        img_smooth_more.save('figures/train/' + key + '/img_smooth_more.png')
        img_gaussian_blur_half.save('figures/train/' + key + '/img_gaussian_blur_half.png')
        img_gaussian_blur_1.save('figures/train/' + key + '/img_gaussian_blur_1.png')
        img_gaussian_blur_1_half.save('figures/train/' + key + '/img_gaussian_blur_1_half.png')
        img_gaussian_blur_2.save('figures/train/' + key + '/img_gaussian_blur_2.png')
        img_modefilter_1.save('figures/train/' + key + '/img_modefilter_1.png')
        img_modefilter_3.save('figures/train/' + key + '/img_modefilter_3.png')
        img_modefilter_5.save('figures/train/' + key + '/img_modefilter_5.png')
        img_minfilter_1.save('figures/train/' + key + '/img_minfilter_1.png')
        img_minfilter_3.save('figures/train/' + key + '/img_minfilter_3.png')
        img_minfilter_5.save('figures/train/' + key + '/img_minfilter_5.png')
        img_unsharpmask.save('figures/train/' + key + '/img_unsharpmask.png')


# def create_dataset():
def main():
    create_and_save_images()


if __name__ == '__main__':
    main()

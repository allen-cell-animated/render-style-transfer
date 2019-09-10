import random
import torchvision.transforms as transforms
import PIL


def lerp(color_0, color_1, alpha):
    return int(alpha * (color_1 - color_0) + color_0)


def get_color(control_point_0, control_point_1, alpha):
    red = lerp(control_point_0[0], control_point_1[0], alpha)
    green = lerp(control_point_0[1], control_point_1[1], alpha)
    blue = lerp(control_point_0[2], control_point_1[2], alpha)
    return (red, green, blue)


def create_map(control_point_1, control_point_2, control_point_3):
    control_point_0 = (0, 0, 0)
    map = [control_point_0]
    stop_0 = 0
    stop_1 = int(255/3)
    stop_2 = int(stop_1 * 2)
    stop_3 = 255
    for i in range(1, 256):
        if (i < stop_1):
            alpha = (i - stop_0) / (stop_1 - stop_0)
            new_color = get_color(control_point_0, control_point_1, alpha)
        elif (i < stop_2):
            alpha = (i - stop_1) / (stop_2 - stop_1)
            new_color = get_color(control_point_1, control_point_2, alpha)
        else: 
            alpha = (i - stop_2) / (stop_3 - stop_2)
            new_color = get_color(control_point_2, control_point_3, alpha)
        map.append(new_color)
    return map


def colorize(image, r1, g1, b1, r2, g2, b2, r3, g3, b3):
    color_filter = PIL.Image.new('RGB', image.size, 'white')
    pixels = color_filter.load()
    lookup_table = create_map((r1, g1, b1),
                              (r2, g2, b2), (r3, g3, b3))
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            # Get Pixel
            pixel = image.getpixel((i, j))
            # Get R, G, B values (This are int from 0 to 255)
            # red = pixel[0] * red_shift
            # green = pixel[1] * green_shift
            # blue = pixel[2] * blue_shift
            # Set Pixel in new image
            intensity = pixel[0]
            pixels[i, j] = lookup_table[intensity]
    return color_filter


def get_random_color():
    return [random.randint(0, 255) for i in range(3)]


def normalize(value, rangetuple):
    return (value - rangetuple[0]) / (rangetuple[1] - rangetuple[0])


class render_function:
    brightness_range = (1.0, 1.5)
    contrast_range = (1.0, 2.0)
    gamma_range = (0.5, 2.0)
    color_range = (0, 255)

    def get_random_params(self, seed):
        random.seed(a=(seed))
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        # hue = random.uniform(-0.5, 0.5)
        # saturation = random.uniform(0.0, 2.0)
        cp_1 = get_random_color()
        cp_2 = get_random_color()
        cp_3 = get_random_color()
        params = [brightness, contrast, gamma, *cp_1, *cp_2, *cp_3]
        return params

    def normalize_render_params(self, render_params):
        colors = [normalize(c, self.color_range) for c in render_params[3:]]
        normalized = [
            normalize(render_params[0], self.brightness_range),
            normalize(render_params[1], self.contrast_range),
            normalize(render_params[2], self.gamma_range),
            *colors
        ]
        return normalized

    def denormalize_render_params(self, normalized_render_params):
        return []  # [normalized_render_params*255]

    # returns a pil-compatible rbg image
    def render(self, input_data, render_params, camera_transform):
        # this function could do anything (e.g. do volume rendering)
        # in this case, the "render" will be a convolution filter and the render params are the filter weights
        # and our camera_transform just rotates the image
        renderedimage = input_data
        # the first 9 render_params need to become a 3x3 filter to apply to the image.
        # if len(render_params) < 9:
        #     raise "Bad number of render_params"
        # convfilter = [[render_params[i + j * 3] for i in range(3)] for j in range(3)]

        # note that groups=3 and we only put one in the array here
        # this is reshaping to make the conv2d happy
        # renderedimage = F.conv2d(
        #     torch.Tensor([input_data]),
        #     # apply same filter to each of r,g,b channels
        #     torch.Tensor([reshaped_params, reshaped_params, reshaped_params]),
        #     bias=None,
        #     stride=1,
        #     padding=1,
        #     dilation=1,
        #     groups=3,
        # )
        renderedimage = transforms.functional.adjust_brightness(
            renderedimage, render_params[0])
        renderedimage = transforms.functional.adjust_contrast(
            renderedimage, render_params[1])

        renderedimage = colorize(
            renderedimage, *render_params[-9:])

        renderedimage = transforms.functional.adjust_gamma(
            renderedimage, render_params[2])
        # renderedimage = transforms.functional.adjust_hue(
        #     renderedimage, render_params[3])
        # renderedimage = transforms.functional.adjust_saturation(
        #     renderedimage, render_params[4])
        

        
        # # normalize back to displayable range after convolution
        # maxval = renderedimage.max()
        # renderedimage = transforms.functional.normalize(
        #     renderedimage[0], mean=[0, 0, 0], std=[maxval, maxval, maxval]
        # )
        # renderedimage_pil = transforms.functional.to_pil_image(renderedimage, mode=None)

        final_image = camera_transform(renderedimage)

        return final_image

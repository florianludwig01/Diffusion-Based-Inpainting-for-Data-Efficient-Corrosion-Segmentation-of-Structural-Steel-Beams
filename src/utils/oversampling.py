import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path


SYNTH_IMAGE_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/"
    "03_Datasets/05_Paper/03_Synth_FLUX/images"
)

SYNTH_LABEL_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/"
    "03_Datasets/05_Paper/03_Synth_FLUX/labels"
)

OUT_IMAGE_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/"
    "03_Datasets/05_Paper/03_Synth_FLUX/02_Synth_FLUX_x10/images"
)

OUT_LABEL_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/"
    "03_Datasets/05_Paper/03_Synth_FLUX/02_Synth_FLUX_x10/labels"
)

OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

OVERSAMPLE_FACTOR = 10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

RANDOM_SEED = 42


def ensure_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def random_brightness_contrast(image):
    alpha = random.uniform(0.65, 1.45)
    beta = random.uniform(-45, 45)

    result = image.astype(np.float32) * alpha + beta
    return ensure_uint8(result)


def random_gamma(image):
    gamma = random.uniform(0.55, 1.8)

    lookup_table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8,
    )

    return cv2.LUT(image, lookup_table)


def random_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_shift = random.uniform(-8, 8)
    saturation_scale = random.uniform(0.65, 1.45)
    value_scale = random.uniform(0.75, 1.25)

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 2] *= value_scale

    hsv = ensure_uint8(hsv)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_channel_shift(image):
    shifts = np.array(
        [
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-20, 20),
        ],
        dtype=np.float32,
    )

    result = image.astype(np.float32) + shifts
    return ensure_uint8(result)


def random_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    clip_limit = random.uniform(1.2, 3.5)
    grid_size = random.choice([4, 6, 8, 10])

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )

    lightness = clahe.apply(lightness)

    result = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def random_gaussian_noise(image):
    sigma = random.uniform(2.0, 18.0)

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    ).astype(np.float32)

    result = image.astype(np.float32) + noise
    return ensure_uint8(result)


def random_poisson_noise(image):
    image_float = image.astype(np.float32) / 255.0
    strength = random.uniform(15.0, 80.0)

    noisy = np.random.poisson(image_float * strength) / strength
    noisy *= 255.0

    return ensure_uint8(noisy)


def random_salt_pepper_noise(image):
    result = image.copy()

    probability = random.uniform(0.0002, 0.003)
    height, width = result.shape[:2]

    number_of_pixels = int(height * width * probability)

    if number_of_pixels <= 0:
        return result

    y_coordinates = np.random.randint(0, height, number_of_pixels)
    x_coordinates = np.random.randint(0, width, number_of_pixels)
    result[y_coordinates, x_coordinates] = 255

    y_coordinates = np.random.randint(0, height, number_of_pixels)
    x_coordinates = np.random.randint(0, width, number_of_pixels)
    result[y_coordinates, x_coordinates] = 0

    return result


def random_blur(image):
    blur_type = random.choice(
        [
            "gaussian",
            "median",
            "average",
            "motion",
        ]
    )

    if blur_type == "gaussian":
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.2, 2.0)

        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            sigma,
        )

    if blur_type == "median":
        kernel_size = random.choice([3, 5])
        return cv2.medianBlur(image, kernel_size)

    if blur_type == "average":
        kernel_size = random.choice([2, 3, 5])
        return cv2.blur(
            image,
            (kernel_size, kernel_size),
        )

    kernel_size = random.choice([3, 5, 7, 9])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    if random.random() < 0.5:
        kernel[kernel_size // 2, :] = 1.0
    else:
        kernel[:, kernel_size // 2] = 1.0

    kernel /= kernel_size

    return cv2.filter2D(image, -1, kernel)


def random_sharpen(image):
    strength = random.uniform(0.3, 1.2)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)

    sharpened = cv2.addWeighted(
        image,
        1.0 + strength,
        blurred,
        -strength,
        0,
    )

    return ensure_uint8(sharpened)


def random_jpeg_compression(image):
    quality = random.randint(35, 92)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        return image

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        return image

    return decoded


def random_vignette(image):
    height, width = image.shape[:2]

    sigma_x = random.uniform(width * 0.35, width * 0.8)
    sigma_y = random.uniform(height * 0.35, height * 0.8)

    kernel_x = cv2.getGaussianKernel(width, sigma_x)
    kernel_y = cv2.getGaussianKernel(height, sigma_y)

    mask = kernel_y @ kernel_x.T
    mask /= mask.max()

    strength = random.uniform(0.15, 0.55)

    if random.random() < 0.85:
        mask = (1.0 - strength) + strength * mask
    else:
        mask = 1.0 + strength * (1.0 - mask)

    result = image.astype(np.float32) * mask[:, :, np.newaxis]
    return ensure_uint8(result)


def random_shadow(image):
    height, width = image.shape[:2]

    shadow_mask = np.zeros((height, width), dtype=np.float32)

    x1 = random.randint(-width // 2, width)
    x2 = random.randint(0, width + width // 2)

    polygon = np.array(
        [
            [x1, 0],
            [x2, 0],
            [
                min(width - 1, x2 + random.randint(-width // 3, width // 3)),
                height - 1,
            ],
            [
                max(0, x1 + random.randint(-width // 3, width // 3)),
                height - 1,
            ],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(shadow_mask, [polygon], 1.0)

    blur_size = max(21, int(min(height, width) * 0.15))
    if blur_size % 2 == 0:
        blur_size += 1

    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (blur_size, blur_size),
        0,
    )

    darkness = random.uniform(0.15, 0.5)
    multiplier = 1.0 - darkness * shadow_mask

    result = image.astype(np.float32) * multiplier[:, :, np.newaxis]
    return ensure_uint8(result)


def random_light_gradient(image):
    height, width = image.shape[:2]

    if random.random() < 0.5:
        gradient = np.linspace(
            random.uniform(0.65, 1.0),
            random.uniform(1.0, 1.35),
            width,
            dtype=np.float32,
        )

        gradient = np.tile(gradient, (height, 1))

    else:
        gradient = np.linspace(
            random.uniform(0.65, 1.0),
            random.uniform(1.0, 1.35),
            height,
            dtype=np.float32,
        )

        gradient = np.tile(gradient[:, np.newaxis], (1, width))

    if random.random() < 0.5:
        gradient = np.flip(gradient, axis=random.choice([0, 1]))

    result = image.astype(np.float32) * gradient[:, :, np.newaxis]
    return ensure_uint8(result)


def random_resolution_degradation(image):
    height, width = image.shape[:2]

    scale = random.uniform(0.35, 0.85)

    small_width = max(2, int(width * scale))
    small_height = max(2, int(height * scale))

    interpolation_down = random.choice(
        [
            cv2.INTER_AREA,
            cv2.INTER_LINEAR,
        ]
    )

    interpolation_up = random.choice(
        [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_NEAREST,
        ]
    )

    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=interpolation_down,
    )

    return cv2.resize(
        small,
        (width, height),
        interpolation=interpolation_up,
    )
    
def augment_image(image):

    aug = image.copy()

    operations = [
        random_brightness_contrast,
        random_gamma,
        random_hsv,
        random_channel_shift,
        random_clahe,
        random_gaussian_noise,
        random_poisson_noise,
        random_salt_pepper_noise,
        random_blur,
        random_sharpen,
        random_jpeg_compression,
        random_vignette,
        random_shadow,
        random_light_gradient,
        random_resolution_degradation,
    ]

    number_of_operations = random.randint(2, 6)
    selected_operations = random.sample(
        operations,
        k=number_of_operations,
    )

    random.shuffle(selected_operations)

    for operation in selected_operations:
        aug = operation(aug)

    return ensure_uint8(aug)
    
def save_image(output_path, image):
    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    if suffix == ".png":
        return cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

    return cv2.imwrite(str(output_path), image)

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    if not SYNTH_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found: {SYNTH_IMAGE_DIR}"
        )

    if not SYNTH_LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Directory not fount: {SYNTH_LABEL_DIR}"
        )

    image_files = sorted(
        p
        for p in SYNTH_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    processed_images = 0
    missing_labels = 0
    failed_images = 0
    written_images = 0

    for image_number, img_path in enumerate(image_files, start=1):
        label_path = SYNTH_LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"KLabel not found: {label_path}")
            missing_labels += 1
            continue

        image = cv2.imread(
            str(img_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(f"Image could not be loaded: {img_path}")
            failed_images += 1
            continue
            
        original_image_path = (
            OUT_IMAGE_DIR
            / f"{img_path.stem}_orig{img_path.suffix.lower()}"
        )

        original_label_path = (
            OUT_LABEL_DIR
            / f"{img_path.stem}_orig.txt"
        )

        if save_image(original_image_path, image):
            shutil.copy2(label_path, original_label_path)
            written_images += 1
        else:
            print(
                "Origianl image could not be saved:",
                original_image_path,
            )

        for augmentation_number in range(OVERSAMPLE_FACTOR):
            augmented_image = augment_image(image)

            output_stem = (
                f"{img_path.stem}_aug{augmentation_number:02d}"
            )

            augmented_image_path = (
                OUT_IMAGE_DIR
                / f"{output_stem}{img_path.suffix.lower()}"
            )

            augmented_label_path = (
                OUT_LABEL_DIR
                / f"{output_stem}.txt"
            )

            success = save_image(
                augmented_image_path,
                augmented_image,
            )

            if not success:
                print(
                    "Augmentated image could not be saved:",
                    augmented_image_path,
                )
                continue

            shutil.copy2(
                label_path,
                augmented_label_path,
            )

            written_images += 1

        processed_images += 1

        if image_number % 50 == 0 or image_number == len(image_files):
            print(
                f"Progress: {image_number}/{len(image_files)} "
            )

if __name__ == "__main__":
    main()

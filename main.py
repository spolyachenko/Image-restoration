import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
import time
import os
import imageio


np.set_printoptions(suppress=True)

path = "example_4.png"

def dirac_like_image(kernel_sz=65):
    res = np.zeros((kernel_sz, kernel_sz))
    res[kernel_sz // 2, kernel_sz // 2] = 255
    return res

def gaussian_psf(k=5, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt

def sum_of_sq_errors_with_tv(img, psf, est_img, alpha = 0):
    psf = psf/psf.sum()

    sq_err = np.abs(img - cv2.filter2D(src=est_img, ddepth=-1, kernel=psf, borderType=cv2.BORDER_REPLICATE))
    sq_err = np.sum(np.power(sq_err, 2))/2

    dxy = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    dxy = cv2.filter2D(src=est_img, ddepth=-1, kernel=dxy, borderType=cv2.BORDER_REPLICATE)
    reg = np.power(np.abs(dxy), 2)
    reg = reg.sum()

    return sq_err + alpha*reg

def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def motion_psf(angle, delta, is_centered=True, kernel_sz=65):
    kernel = np.ones((1, delta), np.float32)
    if not is_centered:
        delta *= 2
        kernel = np.array([0. if i < delta / 2 else 1. for i in range(delta)]).reshape((1, delta))
    h, w = np.shape(kernel)
    kernel = np.pad(kernel, (((kernel_sz - h) // 2, (kernel_sz - h + 1) // 2), ((kernel_sz - w) // 2, (kernel_sz - w + 1) // 2)))
    affine_matrix = cv2.getRotationMatrix2D((kernel_sz//2, kernel_sz//2), angle, 1)
    kernel = cv2.warpAffine(kernel, affine_matrix, (kernel_sz, kernel_sz), flags=cv2.INTER_CUBIC)
    # cv2.imshow("123", kernel1)
    return np.abs(kernel)

def blur_image(input_image, deform_type=None, std_dev=2, delta=8, angle=45, kernel_sz=65):
    img_rgb = input_image
    psf = np.array([])
    if deform_type == "motion":
        if kernel_sz % 2 == 0:
            kernel_sz += 1
        psf = motion_psf(angle, delta, False, kernel_sz)
        cv2.imshow("motion psf", psf)
        psf /= psf.sum()
    elif deform_type == "blur":
        if kernel_sz % 2 == 0:
            kernel_sz += 1
        psf = cv2.GaussianBlur(dirac_like_image(kernel_sz), (kernel_sz, kernel_sz), std_dev)
        cv2.imshow("blur psf", psf)
        psf /= psf.sum()
    else:
        print("Unrecognised deformation type or None specified")
        return None

    res_img = cv2.filter2D(src=img_rgb, ddepth=-1, kernel=psf, borderType=cv2.BORDER_REPLICATE)
    # res_img = add_gaussian_noise(res_img, 0.05)
    return res_img

def regularized_filter(img_rgb, input_psf, lambda_coeff=0.01):
    h, w, _ = img_rgb.shape
    input_psf = input_psf / input_psf.sum()
    psf = np.float32(np.zeros((h, w)))
    kh, kw = input_psf.shape
    psf[:kh, :kw] = input_psf

    PSF = cv2.dft(psf, flags=cv2.DFT_COMPLEX_OUTPUT)

    img_r = np.float32(img_rgb[..., 0])
    img_g = np.float32(img_rgb[..., 1])
    img_b = np.float32(img_rgb[..., 2])

    IMG_R = cv2.dft(img_r, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_G = cv2.dft(img_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_B = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)

    r_tmp = np.float32([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    r = np.pad(r_tmp, ((0, h-3), (0, w-3)))

    R = np.abs(cv2.dft(r, flags=cv2.DFT_COMPLEX_OUTPUT))

    ipsf = (np.abs(PSF) ** 2).sum(-1)
    lmbd_R = lambda_coeff * ((R).sum(-1))
    tmp = (lmbd_R + ipsf)[..., np.newaxis]

    iPSF = np.conj(PSF) / (ipsf + lmbd_R)[..., np.newaxis]

    RES_R = cv2.mulSpectrums(IMG_R, iPSF, 0)
    RES_G = cv2.mulSpectrums(IMG_G, iPSF, 0)
    RES_B = cv2.mulSpectrums(IMG_B, iPSF, 0)

    res_r = cv2.idft(RES_R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_g = cv2.idft(RES_G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_b = cv2.idft(RES_B, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    res_rgb = np.float32(np.zeros_like(img_rgb))
    res_rgb[..., 0] = res_r
    res_rgb[..., 1] = res_g
    res_rgb[..., 2] = res_b

    res_rgb = np.roll(res_rgb, -kh // 2, 0)
    res_rgb = np.roll(res_rgb, -kw // 2, 1)

    return res_rgb

def wiener_filter(img_rgb, input_psf, K = 0.01):
    input_psf = input_psf / input_psf.sum()
    psf = np.float32(np.zeros_like(img_rgb[..., 0]))
    h, w = input_psf.shape
    psf[:h, :w] = input_psf
    # psf[i_h//2 - h//2: i_h//2 - h//2 + h, i_w//2 - w//2: i_w//2 - w//2 + w] = input_psf
    PSF = cv2.dft(psf, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=h)

    img_r = np.float32(img_rgb[..., 0])
    img_g = np.float32(img_rgb[..., 1])
    img_b = np.float32(img_rgb[..., 2])

    IMG_R = cv2.dft(img_r, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_G = cv2.dft(img_g, flags=cv2.DFT_COMPLEX_OUTPUT)
    IMG_B = cv2.dft(img_b, flags=cv2.DFT_COMPLEX_OUTPUT)

    PSF2 = (np.abs(PSF) ** 2).sum(-1)
    iPSF = PSF / (PSF2 + K)[..., np.newaxis]

    RES_R = cv2.mulSpectrums(IMG_R, iPSF, 0)
    RES_G = cv2.mulSpectrums(IMG_G, iPSF, 0)
    RES_B = cv2.mulSpectrums(IMG_B, iPSF, 0)

    res_r = cv2.idft(RES_R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_g = cv2.idft(RES_G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res_b = cv2.idft(RES_B, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    res_rgb = np.float32(np.zeros_like(img_rgb))
    res_rgb[..., 0] = res_r
    res_rgb[..., 1] = res_g
    res_rgb[..., 2] = res_b

    res_rgb = np.roll(res_rgb, -h // 2, 0)
    res_rgb = np.roll(res_rgb, -w // 2, 1)

    return res_rgb

def reg_f_closest_approx(img, psf, st=25, end=26, step=1):
    smallest_error = -1
    best_approx = img
    ind = [u for u in np.arange(st, end, step)]
    errors = []
    for i in np.arange(st, end, step):
        res = regularized_filter(img, psf, 10**(-0.1 * i))
        err = sum_of_sq_errors_with_tv(img, psf, res)
        errors.append(err)
        if err <= smallest_error or smallest_error < 0:
            smallest_error = sum_of_sq_errors_with_tv(img, psf, res)
            best_approx = res

        # res = img_to_0_1_range(res)
        # draw_text(res, "a=" + "{:.6f}".format(10 ** (-0.1 * i)))
        # name = "reg_images\\img" + str(i) + ".jpg"
        # cv2.imwrite(name, res)

    # plt.title("Восстановление при помощи Тихоновской регуляризации")
    # plt.plot(ind, errors)
    # plt.show()

    return best_approx

def wiener_f_closest_approx(img, psf, st=25, end=26, step=1):
    smallest_error = -1
    best_approx = img
    ind = [i for i in np.arange(st, end, step)]
    errors = []
    for i in np.arange(st, end, step):
        res = wiener_filter(img, psf, 10**(-0.1 * i))
        err = sum_of_sq_errors_with_tv(img, psf, res)
        errors.append(err)
        if err <= smallest_error or smallest_error < 0:
            smallest_error = sum_of_sq_errors_with_tv(img, psf, res)
            best_approx = res

        # res = img_to_0_1_range(res)
        # draw_text(res, "K=" + "{:.6f}".format(10 ** (-0.1 * i)))
        # name = "wiener_images\\img" + str(i) + ".jpg"
        # cv2.imwrite(name, res)

    # plt.title("Восстановление при помощи фильтрации Винера")
    # plt.plot(ind, errors)
    # plt.show()

    return best_approx

def simple_lucy_richardson_method(img_rgb, psf, max_iter=80):
    img_r = np.float32(img_rgb[..., 0])
    img_g = np.float32(img_rgb[..., 1])
    img_b = np.float32(img_rgb[..., 2])
    psf = psf/psf.sum()
    psf_t = psf[::-1, ::-1]
    rest_img_r = img_r
    rest_img_g = img_g
    rest_img_b = img_b

    ind = [i for i in range(1, max_iter+1)]
    errors = []

    smallest_err = sum_of_sq_errors_with_tv(img_rgb, psf, img_rgb)
    best_approx = img_rgb
    eps = img_rgb.max() * 1e-5

    for i in range(max_iter):
        tmp_r = img_r / (cv2.filter2D(src=rest_img_r, ddepth=-1, kernel=psf, borderType=cv2.BORDER_REPLICATE) + eps)
        tmp_g = img_g / (cv2.filter2D(src=rest_img_g, ddepth=-1, kernel=psf, borderType=cv2.BORDER_REPLICATE) + eps)
        tmp_b = img_b / (cv2.filter2D(src=rest_img_b, ddepth=-1, kernel=psf, borderType=cv2.BORDER_REPLICATE) + eps)

        tmp_r = cv2.filter2D(src=tmp_r, ddepth=-1, kernel=psf_t, borderType=cv2.BORDER_REPLICATE)
        tmp_g = cv2.filter2D(src=tmp_g, ddepth=-1, kernel=psf_t, borderType=cv2.BORDER_REPLICATE)
        tmp_b = cv2.filter2D(src=tmp_b, ddepth=-1, kernel=psf_t, borderType=cv2.BORDER_REPLICATE)

        rest_img_r = cv2.multiply(rest_img_r, tmp_r)
        rest_img_g = cv2.multiply(rest_img_g, tmp_g)
        rest_img_b = cv2.multiply(rest_img_b, tmp_b)

        tmp_rgb = np.float32(np.zeros_like(img_rgb))
        tmp_rgb[..., 0] = rest_img_r
        tmp_rgb[..., 1] = rest_img_g
        tmp_rgb[..., 2] = rest_img_b

        err = sum_of_sq_errors_with_tv(img_rgb, psf, tmp_rgb)
        errors.append(err)
        if err > smallest_err:
            return tmp_rgb
        if err <= smallest_err:
            smallest_err = err
            best_approx = tmp_rgb

        # tmp_rgb = img_to_0_1_range(tmp_rgb)
        # draw_text(tmp_rgb, str(i))
        # name = "lucy_images\\img" + str(i) + ".jpg"
        # cv2.imwrite(name, tmp_rgb)

    # plt.title("Метод Люси-Ричардсона")
    # plt.plot(ind, errors)
    # plt.show()

    return best_approx

def img_to_0_1_range(img):
    img[img < 0] = 0
    img[img > 1] = 1
    # img = img/np.max(img)
    img = img*255.0
    img = np.round(img).astype(int)
    return img

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_DUPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size

def deform_and_restore(img, deform_type, std_dev=8, delta=17, angle=30, kr_sz=65):
    if deform_type == "motion":
        motion_image = blur_image(img, "motion", std_dev, delta, angle, kr_sz)
        # cv2.imshow("motion blur image", motion_image)

        m_psf = motion_psf(angle, delta, is_centered=False, kernel_sz=kr_sz)

        # start_time = time.time()
        reg_motion_image = reg_f_closest_approx(motion_image, m_psf)
        # cv2.imshow("regularized filter restoration", reg_motion_image)
        # print("--- reg filter %s seconds ---" % (time.time() - start_time))

        # start_time = time.time()
        wiener_motion_image = wiener_f_closest_approx(motion_image, m_psf)
        # cv2.imshow("weiner filter restoration", wiener_motion_image)
        # print("--- wiener filter %s seconds ---" % (time.time() - start_time))

        # start_time = time.time()
        lucy_r_motion_image = simple_lucy_richardson_method(motion_image, m_psf)
        # cv2.imshow("lucy richarson restoration", lucy_r_motion_image)
        # print("--- lucy-rich %s seconds ---" % (time.time() - start_time))

    elif deform_type == "blur":
        blurred_image = blur_image(img, "blur", std_dev, delta, angle, kr_sz)
        cv2.imshow("gaussian blur image", blurred_image)

        b_psf = cv2.GaussianBlur(dirac_like_image(kr_sz), (kr_sz, kr_sz), std_dev)

        reg_blur_image = reg_f_closest_approx(blurred_image, b_psf)
        cv2.imshow("regularized filter restoration", reg_blur_image)

        wiener_blur_image = wiener_f_closest_approx(blurred_image, b_psf)
        cv2.imshow("wiener filter restoration", wiener_blur_image)

        lucy_r_blur_image = simple_lucy_richardson_method(blurred_image, b_psf)
        cv2.imshow("lucy richardson restoration", lucy_r_blur_image)

    else:
        print("Error: wrong deform type")

image = cv2.imread(path, 1)
image = image/255.0
cv2.imshow("original image", image)

deform_and_restore(image, "blur")

# with imageio.get_writer('reg.gif', mode='I') as writer:
#     for i in range(60):
#         filename = "reg_images\\img" + str(i) + ".jpg"
#         image = imageio.imread(filename)
#         writer.append_data(image)
#
# with imageio.get_writer('wiener.gif', mode='I') as writer:
#     for i in range(60):
#         filename = "wiener_images\\img" + str(i) + ".jpg"
#         image = imageio.imread(filename)
#         writer.append_data(image)
#
# with imageio.get_writer('lucy.gif', mode='I') as writer:
#     for i in range(80):
#         filename = "lucy_images\\img" + str(i) + ".jpg"
#         image = imageio.imread(filename)
#         writer.append_data(image)

cv2.waitKey(0)

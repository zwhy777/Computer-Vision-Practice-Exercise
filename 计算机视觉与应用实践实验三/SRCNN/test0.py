import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import Image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from skimage.metrics import structural_similarity as ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="BLAH_BLAH/srcnn_x2.pth")
    parser.add_argument('--image-file', type=str, default="data/123")
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    folder_path = "data/123"
    output_path = 'data/output'

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = pil_image.open(file_path).convert('RGB')
        model = SRCNN().to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()

        image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        #image.save(file_path.replace('.', '_bicubic_x{}.'.format(args.scale)))

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)

        y_cpu = y.cpu()
        preds_cpu = preds.cpu()


        y0 = y_cpu.numpy()
        preds0 = preds_cpu.numpy()
        y0 = np.squeeze(y0)
        preds0 = np.squeeze(preds0)

        ssim_value, _ = ssim(y0, preds0, full=True, data_range=1.0)

        print('SSIM: {:.6f}'.format(ssim_value))

        psnr = calc_psnr(y, preds)
        print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        #output.save(file_path.replace('.', '_srcnn_x{}.'.format(args.scale)))

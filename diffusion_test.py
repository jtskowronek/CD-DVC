"""
1217 Select all test set data for testing and take the average
1229 Test using Konda according to the paper, batch value should be 1
"""
import re
import numpy as np
import torch
import configparser
import os
import time
import math
from image_comp.metric import *
import cv2
import torch.utils.data as data
import image_comp.datasetDistribute0318 as datasetDistribute
from torchvision import transforms
from modules.model import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules import BigCompressor
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize
from PIL import Image
import lpips
from skimage.metrics import peak_signal_noise_ratio as cacula_psnr
# import warnings
# warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = torch.cuda.device_count()

lpips_model = lpips.LPIPS(net='vgg')  # Choose 'vgg' or 'alex'


def clear_folder(folder_path):
    """Delete all files in folder"""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder_path)

def pad_images_to_192x192(input_folder, output_folder):
    """Pad images to 192x192"""
    clear_folder(output_folder)
    print("folder_clear!")
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')  # Open as color image

        # Original image dimensions
        original_width, original_height = img.size

        # Create 192x192 black background image
        new_img = Image.new('RGB', (192, 192), (0, 0, 0))  # Black background

        # Paste original image onto black background image
        offset = ((192 - original_width) // 2, (192 - original_height) // 2)
        new_img.paste(img, offset)

        # Save image
        new_img.save(os.path.join(output_folder, img_name))

    print(f"Images in {input_folder} padded and saved to {output_folder}")


def pad_and_filter_images(img_path, output_folder):
    """Pad images to 192x192 and filter processing"""
    original_size = (176, 144)  # Original image size
    target_size = (192, 192)  # Target image size

    # Process even-numbered images: from 2 to 148, step 2
    for i in range(2, 300, 2):
        file_name = f"hall_qcif_{i:05d}.png"  # Filename format hall_qcif_00002.png to hall_qcif_00148.png

        # Build complete input image path
        input_path = os.path.join(img_path, file_name)

        # Check if file exists
        if not os.path.exists(input_path):
            print(f"File {input_path} does not exist, skipping this file.")
            continue

        # Open image
        img = Image.open(input_path).convert('RGB')  # Open as color image

        # Check if image dimensions are 176x144
        if img.size != original_size:
            print(f"Image {file_name} dimensions do not match, skipping this file.")
            continue

        # Create 192x192 black background image
        new_img = Image.new('RGB', target_size, (0, 0, 0))  # Black background

        # Calculate position to place original image in new image for centering
        x_offset = (target_size[0] - original_size[0]) // 2
        y_offset = (target_size[1] - original_size[1]) // 2

        # Paste the original image to the center of the black background image
        new_img.paste(img, (x_offset, y_offset))

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, file_name)
        new_img.save(output_path)

    print(f"Images in {img_path} padded and saved to {output_folder}")



def crop_and_save_images(input_folder, output_folder):
    """Crop 192x192 color images to 176x144 and save to the target folder"""
    clear_folder(output_folder)
    print("folder_clear!")

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')  # Open as color image

        # Ensure image size is 192x192
        if img.size != (192, 192):
            print(f"Skipping {img_name} because its size is not 192x192")
            continue

        # Calculate crop area
        left = (192 - 176) // 2
        top = (192 - 144) // 2
        right = left + 176
        bottom = top + 144

        # Crop image
        cropped_img = img.crop((left, top, right, bottom))

        # Save image
        cropped_img.save(os.path.join(output_folder, img_name))

    print(f"Images in {input_folder} cropped to 176x144 and saved to {output_folder}")


def file_name(path):
    file_list=[]
    pth = os.listdir(path)
    for name in pth:
        for root, dirs, files in os.walk(os.getcwd()):
            for tt in range(len(files)):
                file_list.append(files[tt]) # All non-directory files in the current path
    return file_list

def gauss_noise(image, mean=0, var=0.001):
    '''
        Add Gaussian noise
        mean : mean value
        var : variance
    '''
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #out = np.uint8(out*255)
    #cv.imshow("gauss", out)
    return out


def get_args(filename):
    args = {}
    config = configparser.RawConfigParser()

    # Use `read_string` to read config file without section
    with open(filename, 'r') as f:
        config.read_string("[DEFAULT]\n" + f.read())

    # Directly process key-value pairs
    for option in config['DEFAULT']:
        value = config['DEFAULT'][option]
        if value.isdigit():
            args[option] = int(value)
        else:
            try:
                args[option] = float(value)
            except ValueError:
                args[option] = value
    return args


def psnr01(img1, img2):
    #mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def load_set(test,batch_size):
    test_set = datasetDistribute.ImageFolder(is_train=False, root=test)
    # 1210 Load test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return test_loader
def flag_judge(flag,imgPre,imgMid,imgNext):
    if flag == 1:  # Previous
        dataSide = imgPre
    elif flag == 2:  # Middle
        dataSide = imgMid
    elif flag == 3:  # Next
        dataSide = imgNext
    elif flag == 4:  # Noise
        dataSide = torch.FloatTensor(gauss_noise(imgMid.numpy()))
    elif flag == 5:  # Previous and next frame
        dataSide = x = torch.cat([imgPre, imgNext], dim=1)
    elif flag == 6:
        dataSide = (imgPre + imgNext) / 2
    return dataSide



def encode_key(filename,QP):
    if os.path.exists('str.bin'):
        os.system('rm str.bin')
        os.system('rm rec.yuv')
        print('delete old str.bin and rec.yuv !')
    
    if os.path.exists('str.bin'):
        print('error')

    if os.path.exists('log.txt'):
        os.system('rm log.txt')
        print('delete old log.txt !')
    print('encodering Key frames ......')
    commd = '/home/test/yezhuang/VVCSoftware_VTM-VTM-16.0/bin/EncoderAppStatic -c key_cfg/encoder_intra_vtm.cfg -c key_cfg/BQSquare.cfg  -i {} -ts 2 -f 149 -q {} > log.txt'.format(filename,QP)
    print('asd:',os.getcwd())
    if os.system(commd)!=0:
        print('encodering Key frames fail !')


    print("Key frames encoder finish !")

def decode_key(de_key_path, height, width, start_frame):
    """
    Extract image frames from YUV file and save as PNG format.
    :param de_key_path: Folder path to save decoded images
    :param height: Image height
    :param width: Image width
    :param start_frame: Start frame
    :return: None
    """
    # Clear target folder
    if os.listdir(de_key_path):
        os.system('rm {}/*'.format(de_key_path))
        print('Key frames folder cleared!')

    yuv_file = 'rec.yuv'
    if os.path.exists(yuv_file):
        # Calculate size per frame (YUV420)
        frame_size = width * height + (width // 2) * (height // 2) * 2  # Y + U + V
        # Open YUV file and calculate total frames
        with open(yuv_file, 'rb') as fp:
            fp.seek(0, 2)  # Set file pointer to end of file
            fp_end = fp.tell()  # Get file end pointer position
            num_frames = fp_end // frame_size  # Calculate number of frames in file

            print(f"This {yuv_file} file has {num_frames} frames!")

            # Set file pointer to start frame
            fp.seek(frame_size * start_frame, 0)
            t = 1
            for i in range(num_frames - start_frame):
                # Read one frame data (YUV420 format)
                yuv_data = np.frombuffer(fp.read(frame_size), dtype=np.uint8)

                # Separate Y, U, V components
                y = yuv_data[:width * height].reshape((height, width))
                u = yuv_data[width * height:width * height + (width // 2) * (height // 2)].reshape(
                    (height // 2, width // 2))
                v = yuv_data[width * height + (width // 2) * (height // 2):].reshape((height // 2, width // 2))

                # Upsample U and V components to same resolution as Y
                u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

                # Merge YUV components and convert to BGR format
                yuv_img = cv2.merge([y, u_up, v_up])
                bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

                # Save image
                output_file = os.path.join(de_key_path, f'hall_qcif_{t:05d}.png')
                cv2.imwrite(output_file, bgr_img)
                #print(f"Saved frame {t} to {output_file}")

                t += 2

        print(f'Key frames decoded to {de_key_path} successfully!')
    else:
        print(f'File {yuv_file} does not exist!')


def yuv2img(file_name, save_path, height, width, start_frame):
    """
    Convert YUV video file to images
    :param file_name: Name of YUV video to process
    :param save_path: Folder path to save images
    :param height: Image height
    :param width: Image width
    :param start_frame: Start frame
    :return: None
    """
    # Clear target folder
    if os.listdir(save_path):
        os.system('rm {}/*'.format(save_path))
        print('origin image frames folder cleared!')

    # Calculate size per frame (YUV420)
    frame_size = width * height + (width // 2) * (height // 2) * 2  # Y + U + V

    # Open file and calculate total frames
    with open(file_name, 'rb') as fp:
        fp.seek(0, 2)  # Set file pointer to end of file
        fp_end = fp.tell()  # Get file end pointer position
        num_frames = fp_end // frame_size  # Calculate number of frames in file

        print(f"This {file_name} file has {num_frames} frame imgs!")

        # Set file pointer to start frame
        fp.seek(frame_size * start_frame, 0)

        for i in range(num_frames - start_frame):
            # Read one frame data (YUV420 format)
            yuv_data = np.frombuffer(fp.read(frame_size), dtype=np.uint8)

            # Separate Y, U, V components
            y = yuv_data[:width * height].reshape((height, width))
            u = yuv_data[width * height:width * height + (width // 2) * (height // 2)].reshape((height // 2, width // 2))
            v = yuv_data[width * height + (width // 2) * (height // 2):].reshape((height // 2, width // 2))

            # Upsample U and V components to same resolution as Y
            u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

            # Merge YUV components and convert to BGR format
            yuv_img = cv2.merge([y, u_up, v_up])
            bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

            # Save image
            cv2.imwrite(f'{save_path}/hall_qcif_{i + 1:05d}.png', bgr_img)

        print(f"Extracted {num_frames - start_frame} frames, saved to {save_path}.")

    print(f"{file_name} convert to image finished!")



def pipei(pth):
    while not os.path.exists(pth):  # Check if file exists
        pth = input('Cann\'t find the file,Please input the correct file pth:')
    data = open(pth, 'r')  # Open file
    flag = 0
    p = re.compile(r'LayerId')
    for lines in data:
        value = lines.split('\t')  # Read each line
        # print(value)
        if flag:
            if flag ==1:
                # print(str(value)[-13:-5])
                psnr = float(str(value)[-13:-5])
                print(psnr)
            flag -= 1
        if re.search(p, str(value)):
            flag = 2
    data.close()
    #print("psrt:::::",psnr)
    return psnr


transform = transforms.ToTensor()

def calculate_psnr(img1_path, img2_path):
    original_img = Image.open(img1_path)
    compressed_img = Image.open(img2_path)

    # Resize to be consistent
    original_img = resize(transform(original_img), (144, 176))
    compressed_img = resize(transform(compressed_img), (144, 176))

    # Calculate PSNR
    # psnr_value = calculate_psnr(original_img.numpy(), compressed_img.numpy())
    return cacula_psnr(original_img.numpy(), compressed_img.numpy())

def calculate_lpips(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    # transform = transforms.ToTensor()
    img1 = transform(img1)
    img2 = transform(img2)
    # Ensure img1 and img2 are 4D tensor (batch, channels, height, width)
    # if len(img1.shape) == 3:
    #     img1 = img1.unsqueeze(0)
    # if len(img2.shape) == 3:
    #     img2 = img2.unsqueeze(0)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    # Calculate LPIPS
    lpips_value = lpips_model(img1, img2)
    return lpips_value.item()

def video_cat(path1,path2,num=300):
    if os.path.exists('result.yuv'):
        os.system('rm result.yuv')
        print('delete result.yuv,and create new result.yuv !')
    fp = open('result.yuv', 'wb+')
    ssim = []
    psnr = []
    lpips = []
    start = time.time()
    for i in range(1, num):
        if i % 2 != 0:
            ssim.append(msssim('/home/test/yezhuang/test_video_vvc/img_path/hall_qcif_%05d.png' % (i),
                               path1 + '/hall_qcif_%05d.png' % (i)))
            #psnr.append(calculate_psnr('/root/yezhuang/test_video_comp_VTM/img_path/hall_qcif_%05d.png' % (i),path1 + '/hall_qcif_%05d.png' % (i)))
            lpips.append(calculate_lpips('/home/test/yezhuang/test_video_vvc/img_path/hall_qcif_%05d.png' % (i),
                                         path1 + '/hall_qcif_%05d.png' % (i)))
            image = Image.open(path1 + '/hall_qcif_%05d.png' % (i))
        else:
            ssim.append(msssim('/home/test/yezhuang/test_video_vvc/img_path/hall_qcif_%05d.png' % (i), path2 + '/hall_qcif_{:05d}.png'.format(i)))
            #psnr.append(calculate_psnr('/root/yezhuang/test_video_comp_VTM/img_path/hall_qcif_%05d.png' % (i),path2 + '/hall_qcif_{:05d}.png'.format(i)))
            lpips.append(calculate_lpips('/home/test/yezhuang/test_video_vvc/img_path/hall_qcif_%05d.png' % (i), path2 + '/hall_qcif_{:05d}.png'.format(i)))
            image = Image.open(path2 + '/hall_qcif_{:05d}.png'.format(i))

        image = np.asarray(image)
        fp.write(image)
    fp.close()
    end = time.time() - start
    print("cat time :", end)
    print('video image SS-SSIM average: ',np.mean(ssim))
    #print('video image PSNR average: ',np.mean(psnr))
    print('video image LPIPS average: ',np.mean(lpips))

    print('Images(WZ and Key frames) merge into result.yuv success !')




def get_bps(path1,path2):
    key_size = os.path.getsize(path1)*8.0
    wz_size = os.path.getsize(path2)*8.0
    bps = (key_size/75.0*7.5+wz_size/74.0*7.5)/1024.0
    return bps

# ################################################################
# ################################################################

def main(rank):
    args = get_args("config.ini")
    path = "/home/test/yezhuang/test_video_vvc/decoder_img"

    if os.listdir(path):
        os.system('rm {}/*'.format(path))
        print('Decoder WZ frames, WZ frames folder clear !!')


    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )
    denoise_model_cor = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )
    context_model = BigCompressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )
    context_model_cor = BigCompressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )
    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        denoise_fn_cor=denoise_model_cor,
        context_fn=context_model,
        context_fn_cor=context_model_cor,
        num_timesteps=20000,
        clip_noise="none",
        vbr=False,
        lagrangian=0.9,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=args['lpips_weight'],
        aux_loss_type="lpips"
    )

    loaded_param = torch.load(
        args['ckpt'],
        map_location=lambda storage, loc: storage,
    )

    diffusion.load_state_dict(loaded_param["model"])
    diffusion.to(rank)
    diffusion.eval()

    #yuv2img(args['filename'], args['img_path'], 144, 176, 0)
    #encode_key(args['filename'], args['key_qp'])
    #decode_key(args['de_key_path'], 144, 176, 0)
    #rename_and_move_images(args['temp_de_key'],args['de_key_path'] )

    # First process images in de_key_path
    pad_images_to_192x192(args['de_key_path'], args['padded_img_path'])
    # Then process images in img_path
    pad_and_filter_images(args['img_path'], args['padded_img_path'])
    test_set = datasetDistribute.ImageFolder(is_train=False, root=args['padded_img_path'])
    # 1210 Load test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    key_size = os.path.getsize("/home/test/yezhuang/work/video/300frames/hall/qp_24/str_hall_176x144_15fps_420_8bit_YUV_24.bin") * 8.0
    key_bps = (key_size / 150.0 * 7.5 ) / 1024.0
    print("key bps:",key_bps)
    total_bpp = 0
    num = []

    start = time.time()
    for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(test_loader):
        if batch >= 300:
            break
        if batch % 2 == 0:
            continue
        else :
            imgPre = imgAll[:, 0:3, :, :]
            imgMid = imgAll[:, 3:6, :, :]
            imgNext = imgAll[:, 6:9, :, :]

            print("decode WZ: {}".format(batch+1))
            data1 = imgMid
            dataSide = (imgPre + imgNext) / 2
            data1 = data1.cuda()
            dataSide = dataSide.cuda()
            #print("data1: ", data1.shape)
            #print("dataSide: ", dataSide.shape)
            compressed_x, bpp, transmitted_bpp = diffusion.compress(
                data1 * 2.0 - 1.0,
                dataSide * 2.0 - 1.0,
                sample_steps=200,
                sample_mode="ddim",
                bpp_return_mean=False,
                init=torch.randn_like(data1) * 0.8
            )

            total_bpp += bpp.mean().item()
            print("bpp: ", bpp)
            compressed = compressed_x.clamp(-1, 1) / 2.0 + 0.5
            compressed_croped = TF.center_crop(compressed,(144,176))
            data1 = TF.center_crop(data1,(144,176))
            #print(compressed)
            torchvision.utils.save_image(compressed_croped.cpu(), os.path.join(path, "hall_qcif_{:05d}.png".format(batch+1)))
            psnr = psnr01(compressed_croped.cpu().detach().numpy(), data1.cpu().detach().numpy())
            print("psnr:",psnr)
            num.append(psnr01(compressed_croped.cpu().detach().numpy(), data1.cpu().detach().numpy()))
    wholetime = time.time()-start
    print("time: ", wholetime)
    wz_bpp = total_bpp/150
    print(wz_bpp)
    wz_bps = wz_bpp*176*144*7.5/1024
    bps = key_bps + wz_bps
    #crop_and_save_images(path, args['de_key_path'])
    #crop_and_save_images(path, args['recon_file'])
    print("encoder bitrate : {} kbps ".format(bps));
    print('video all average psnr :',
          (np.mean(num) + \
              pipei(
                  '/home/test/yezhuang/work/video/300frames/hall/qp_24/log_hall_176x144_15fps_420_8bit_YUV_24.txt'
                  )
              ) / 2.0
          )

    video_cat(args['de_key_path'], args['output_file'])

if __name__ == "__main__":
    args = get_args("config.ini")
    main(args['device'])

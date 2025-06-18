"""
1217 选择测试集的所有数据进行测试，取平均值
1229 按照论文用康达进行测试  batch值应为1
"""
import argparse
import sys
import numpy as np
import torch
from torch.autograd import Variable
import configparser
import os
import math
from .metric import *
import imageio
import torch.utils.data as data
from . import datasetDistribute0318 as datasetDistribute
from . import networkDistribute_pyramid as networkDistribute
from torchvision import transforms
from . import analysis_side as side_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = torch.cuda.device_count()
def file_name(path):
    file_list=[]
    dir = os.listdir(path)
    for name in dir:
        for root, dirs, files in os.walk(os.getcwd()):
            for tt in range(len(files)):
                file_list.append(files[tt]) #当前路径下所有非目录子文件
    return file_list
def gauss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
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
def get_args(filename,section):
    args = {}
    config = configparser.RawConfigParser()
    config.read(filename)
    options = config.options(section)
    print(len(options))
    for t in range(len(options)):
        if config.get(section,options[t] ).isdigit():
            args[options[t]] = int(config.get(section,options[t] ))
        else:
            try:
                float(config.get(section,options[t] ))
                args[options[t]] = float(config.get(section,options[t] ))
            except:
                args[options[t]] = config.get(section, options[t])
    return args

def init_lstm_test(name,batch_size, height, width):
    """
    1204 添加  用函数替换
    """
    if name == 'encoder':
        with torch.no_grad():
            encoder_h_1 = (Variable(torch.zeros(batch_size, 128, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128, height // 4, width // 4)))
            encoder_h_2 = (Variable(torch.zeros(batch_size, 256, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256, height // 8, width // 8)))
            encoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256, height // 16, width // 16)))

            decoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256, height // 16, width // 16)))
            decoder_h_2 = (Variable(torch.zeros(batch_size, 256, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256, height // 8, width // 8)))
            decoder_h_3 = (Variable(torch.zeros(batch_size, 128, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128, height // 4, width // 4)))
            decoder_h_4 = (Variable(torch.zeros(batch_size, 64, height // 2, width // 2)),
                           Variable(torch.zeros(batch_size, 64, height // 2, width // 2)))

            encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
            encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
            encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

        return (encoder_h_1, encoder_h_2, encoder_h_3,
                decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
    if name == 'decoder':
        with torch.no_grad():    

            decoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256, height // 16, width // 16)))
            decoder_h_2 = (Variable(torch.zeros(batch_size, 256, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256, height // 8, width // 8)))
            decoder_h_3 = (Variable(torch.zeros(batch_size, 128, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128, height // 4, width // 4)))
            decoder_h_4 = (Variable(torch.zeros(batch_size, 64, height // 2, width // 2)),
                           Variable(torch.zeros(batch_size, 64, height // 2, width // 2)))


            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

        return (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
def psnr01(img1, img2):
    #mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def load_model(model):
    encoder.load_state_dict(torch.load(model))
    binarizer.load_state_dict(torch.load(model.replace('encoder', 'binarizer')))
    decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder')))
    return encoder,binarizer,decoder
def load_model_side(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0
def load_set(test,batch_size):
    test_set = datasetDistribute.ImageFolder(is_train=False, root=test)
    # 1210 加载test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=1)
    return test_loader
def flag_judge(flag,imgPre,imgMid,imgNext):
    if flag == 1:  # 前
        dataSide = imgPre
    elif flag == 2:  # 中
        dataSide = imgMid
    elif flag == 3:  # 后
        dataSide = imgNext
    elif flag == 4:  # 噪声
        dataSide = torch.FloatTensor(gauss_noise(imgMid.numpy()))
    elif flag == 5:  # 前一帧和后一帧
        dataSide = x = torch.cat([imgPre, imgNext], dim=1)
    elif flag == 6:
        dataSide = (imgPre + imgNext) / 2
    return dataSide

def encoder_img(train_dataset,max_batch,encoder,binarizer,flag,level,iterations,output_name):
    test_set = datasetDistribute.ImageFolder(is_train=False, root=train_dataset)
    # 1210 加载test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    file = []
    for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(test_loader):

        print('filename: ',filename,filenamePre, filenameNext)
        encoder.eval()
        binarizer.eval()
        # decoder.eval()

        imgPre = imgAll[:, 0:1, :, :]
        imgMid = imgAll[:, 1:2, :, :]
        imgNext = imgAll[:, 2:3, :, :]
        data1 = imgMid

        dataSide = flag_judge(flag,imgPre,imgMid,imgNext)
        with torch.no_grad():
            image = Variable(data1.cuda())
        dataSide = Variable(dataSide.cuda())

        res = image - 0.5
        dataSide = dataSide - 0.5

        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm_test(
            name = 'encoder',batch_size=image.size(0), height=image.size(2), width=image.size(3))
        codes = []
        for iters in range(iterations):
            # Encode.
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            # Binarize.
            code = binarizer(encoded)
           # print(code.shape)
            # Decode.
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, dataSide)

            res = res - output  # Variable

            codes.append(code.data.cpu().numpy())
            
        # codes = (np.stack(codes).astype(np.int8) + 1) // 2
        #
        # export = np.packbits(codes.reshape(-1))
        file.append(codes)
        # np.savez_compressed(output_name+str(batch), shape=codes.shape, codes=export)
        print(batch)
    codes = (np.stack(file).astype(np.int8) + 1) // 2
    export = np.packbits(codes.reshape(-1))
    np.savez_compressed(output_name+'all', shape=codes.shape, codes=export)
def decoder_img(input_file,input_name,tframe,kframe,max_batch,encoder,binarizer,flag,level,iterations,output_file,output_name):

    test_set = datasetDistribute.ImageFolder(is_train=False, root=kframe)
    # 1210 加载test
    kframe_loader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    
    test_set0 = datasetDistribute.ImageFolder(is_train=False, root=tframe)
        # 1210 加载test
    #tframe_loader = data.DataLoader(
    #    dataset=test_set0, batch_size=1, shuffle=False, num_workers=1)
    num = []
    content = np.load(input_file + '/output_img' + 'all' + '.npz')
    codes_all = np.unpackbits(content['codes'])
    codes_all = np.reshape(codes_all, content['shape']).astype(np.float32) * 2 - 1

    codes_all = torch.from_numpy(codes_all)

    for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(kframe_loader):
        print('encoder:output_img'+str(batch)+'.npz')

        codes = codes_all[batch]
        iters, batch_size1, channels, height, width = codes.size()
        height = height * 16
        width = width * 16
        with torch.no_grad():
            codes = Variable(codes)
        codes = codes.cuda()
        # decoder.eval()
        (imgAll0, filename0, filenamePre0, filenameNext0) = test_set0[batch]
        #print(filename0)
        imgMid0 = imgAll0[ 1:2, :, :]


        imgPre = imgAll[:, 0:1, :, :]
        imgMid = imgAll[:, 1:2, :, :]
        imgNext = imgAll[:, 2:3, :, :]
        #data = imgMid

        #dataSide = imgPre
        dataSide = flag_judge(flag,imgPre,imgMid,imgNext)
        
        dataSide = Variable(dataSide.cuda())    
        #dataSide = Variable(dataSide)
        dataSide = dataSide - 0.5

        (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm_test(name = 'decoder',
                batch_size=batch_size1, height=height, width=width)
        #print('imgPre.shape',imgPre.shape)
        #print('code.shape',codes.shape)
       
        image = torch.zeros(imgPre.shape) + 0.5  # 确定输出的尺寸 1204添加
        image = Variable(image)  # 转化成相同的数据类型 1204
        print('length:',len(codes))
        for iters in range(min(iterations, codes.size(0))):
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, dataSide)
            image = image + output.data.cpu()
            #print('iters',iters)
        #image = image -0.5
        print('write: '+output_file+'/'+output_name+str(batch)+'.png')
        
        input = torch.cat((imgPre, image.cpu(),imgNext), dim=1)   
        recon_image = net_side(input)
        print('PSNR:',psnr01(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        num.append(psnr01(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        
        imageio.imwrite(
            os.path.join(output_file+'/'+output_name+'{:02d}.png'.format(batch)),
            np.squeeze(recon_image.cpu().detach().numpy().clip(0, 1) * 255.0).astype(np.uint8))
    print("Average PSNR：",np.mean(num))    


encoder = networkDistribute.EncoderCell().cuda()
binarizer = networkDistribute.Binarizer().cuda()
decoder = networkDistribute.DecoderCell().cuda()

model_side = side_net.Analysis_side_net(32, 64)

# torch.load 加载模型
# replace 替换字符串
#encoder.load_state_dict(torch.load(args['model']))
#binarizer.load_state_dict(torch.load(args['model'].replace('encoder', 'binarizer')))
#decoder.load_state_dict(torch.load(args['model'].replace('encoder', 'decoder')))

if(sys.argv[1] == 'encoder'):
    args = get_args("config.ini",'encoder')
    load_model(args['model'])
    encoder_img(args['test'],args['max_batch'],encoder,binarizer,args['flag'],args['level'],args['iterations'],args['output_name'])
elif (sys.argv[1] == 'decoder'):
    args = get_args("config.ini",'decoder')
    load_model(args['model'])
    global_step = load_model_side(model_side, args['sidetrain'])
    net_side = model_side.cuda()
    net_side = torch.nn.DataParallel(net_side, list(range(gpu_num)))
    decoder_img(args['input_file'],'output_img',args['test0'],args['test'],args['max_batch'],encoder,binarizer,args['flag'],args['level'],args['iterations'],args['output_file'],args['output_name'])
else:
    print(sys.argv[1])
    print("please input a parameter (encoder or decoder)")
    exit()

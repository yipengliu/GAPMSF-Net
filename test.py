from dataset import *
from GAP_GRUacc import Tensor_AMP_net
#from nodeblock_model import Tensor_AMP_net
#from models import Tensor_AMP_net
import numpy as np
from torch.autograd import Variable
import math
from PIL import Image
from utlis import *
import os
import h5py
from time import time
import scipy
#import pynvml
from skimage.morphology import disk
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_sampling_matrix():
    #path = "maskData/mask_real_10_512_new.mat"
    path = "/home/hengling/lx/TensorAMP_Net/maskData/mask256.mat" #mask256
    mask = io.loadmat(path)["phi"]#phi
    #print(mask.shape)
    return mask

def train(model, opt, train_loader, epoch, batch_size, PhaseNum):
    model.train()
    n = 0
    for data in train_loader:
        n = n + 1
        opt.zero_grad()  # 清空梯度
        # data = torch.unsqueeze(data,dim=1), torch.unsqueeze(target,dim=1)/
        data = Variable(data.float().cuda())

        outputs = model(data, PhaseNum)

        # loss_all = compute_loss(outputs,data)
        # loss = get_final_loss(loss_all)
        # loss = torch.mean((outputs[-1]-target)**2)
        loss = torch.mean((outputs - data) ** 2)
        loss.backward()
        opt.step()
        if n % 5 == 0:
            output = "PhaseNum: %d [%02d/%02d] loss: %.8f" % (
            PhaseNum, epoch, batch_size * n, loss.data.item())
            # output = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch, batch_size*n,
            #                                        cost.data.item(),cost_sym.data.item())
            print(output)
def save_to_mat(img, output_name):
    new_data_path = os.path.join(os.getcwd(), "aaa/big")
    if not os.path.isdir(new_data_path):
        os.mkdir(new_data_path)
    npy_data = np.array(img, dtype="uint16")
    #print(npy_data.shape)
    np.save(new_data_path + '/{}.npy'.format(output_name), npy_data)
    npy_load = np.load(new_data_path + '/{}.npy'.format(output_name))
    io.savemat(new_data_path + '/{}.mat'.format(output_name), {'data': npy_load})
## 这一点待定，因为需要确定如何测量，即弄清楚别人论文里面的评价指标
def get_val_result(model, num, is_cuda=True):
    model.eval()
    imagesize= 256
    frame = 2
    test_set_path = "val.pt"
    data_path = "/home/hengling/lx/TensorAMP_Net/test_data/traffic_cacti.mat"
    #data_path = "new_test/pendulumBall_10.mat"
    #data_path = "new_test/mat/red_256/red_big_R.mat"#################################
    #data = io.loadmat(data_path)["pic"]
    
    flag=5
    if flag == 2:
        data = io.loadmat(data_path)["labels"]#["labels"] orig Kobe park
        
        
        data_input = torch.Tensor(data)
        #print(data_input.shape)
        #data_input = data_input.reshape(6,8,256,256)
        data_input = data_input.permute(0,3,1,2)
        #print(data_input.shape)
        #data_input = data_input[2:4,:,:,:]
        print(data_input.shape)
    elif flag ==4:#真实数据  注意输入
        data = io.loadmat(data_path)["meas"]#["labels"] 
        #print(data)
        data_input = data[:,:,4:5]/255
        data_input = np.expand_dims(data_input,2)#/255
        data_input = torch.Tensor(data_input)
        print(data_input.shape)
        data_input = data_input.permute(3,2,0,1)#.reshape(2,1,256,256)
        print(data_input.shape)
    else:##drop traffic runner
        data = io.loadmat(data_path)["orig"]#["labels"]
        data_input = torch.Tensor(data)
        data_input = torch.unsqueeze(data_input.permute(2,0,1),0)/255
        data_input = data_input.reshape(6,8,256,256)
        data_input =data_input[0:4,:,:,:]
    if flag==4:
        data_input = Variable(data_input.float().cuda())
        outputs = model(data_input,num)
        print(outputs.shape)
        outputs = outputs.permute(2,3,0,1)
        print(outputs.shape)
        outputs = torch.reshape(outputs, [imagesize,imagesize,1*10])

        ImgNum =32 # 测试图像的数量
        for index in range(ImgNum):
            fake_one = outputs[:,:,index]*255
            if is_cuda:
                fake_one = fake_one.cpu().data.numpy()
            else:
                fake_one = fake_one.data.numpy()
            
            fake_one = fake_one.astype(int)
            bbb = fake_one < 0
            fake_one[bbb] = 0###
            bbb = fake_one > 255
            fake_one[bbb] = 255
            
            print(fake_one.shape)
            #savedata[index,:,:] = np.expand_dims(fake_one,0)
            im = Image.fromarray(np.uint8(fake_one))
            path = str(index+8)+".png"
            im.save("gif/amp/"+path)
        im = Image.open("gif/amp/0.png")
        images=[]
        for i in range(10):
            fpath = "gif/amp/"+ str(i) + ".png"
            images.append(Image.open(fpath))
        im.save("drop.gif", save_all=True, append_images=images,loop=100,duration=0.1)
    else:
        with torch.no_grad():
            data_input = Variable(data_input.float().cuda())
            #print(data_input.shape)
            start = time()
            outputs= model(data_input,num)
            end = time()
            print("time:",end-start)
            #print(time)
            #outputs = torch.squeeze(outputs,0)
            #print(111111111111) 


            data_input = data_input.permute(2,3,0,1)

            outputs = outputs.permute(2,3,0,1)
            data_input = torch.reshape(data_input,[imagesize, imagesize, 4*8])
            outputs = torch.reshape(outputs, [imagesize,imagesize,4*8])

            data_input = data_input.permute(2,0,1)
            outputs = outputs.permute(2,0,1)
            ImgNum =32 # 测试图像的数量
            PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
            output_name = "red_big_R"########################################################
            #savedata = np.zeros((8, imagesize, imagesize))
        for index in range(ImgNum):
            real_one = data_input[index,:,:]*255
            fake_one = outputs[index,:,:]*255
            if is_cuda:
                fake_one = fake_one.cpu().data.numpy()
                real_one = real_one.cpu().data.numpy()
            else:
                fake_one = fake_one.data.numpy()
                real_one = real_one.data.numpy()
            
            fake_one = fake_one.astype(int)
            bbb = fake_one < 0
            fake_one[bbb] = 0###
            bbb = fake_one > 255
            fake_one[bbb] = 255
            #print(fake_one.shape)
            #savedata[index,:,:] = np.expand_dims(fake_one,0)
            im = Image.fromarray(np.uint8(fake_one))
            path = str(index)+".png"################################################
            im.save("gif/amp/"+path)
            
            real_one = real_one.astype(int)
            bbb = real_one < 0
            real_one[bbb] = 0
            bbb = real_one > 255
            real_one[bbb] = 255

            rec_PSNR = psnr(fake_one, real_one)  # 计算PSNR的值
            #print(rec_PSNR)
            PSNR_All[0, index] = rec_PSNR
            rec_SSIM = compute_ssim(fake_one, real_one)  # 计算PSNR的值
            SSIM_All[0, index] = rec_SSIM
        #save_to_mat(savedata, output_name)
        return np.mean(PSNR_All),np.mean(SSIM_All)
        """
        im = Image.open("gif/amp/0.png")
        images=[]
        for i in range(20):
            fpath = "gif/amp/"+ str(i) + ".png"
            images.append(Image.open(fpath))
        im.save("drop.gif", save_all=True, append_images=images,loop=100,duration=0.1)
        return np.mean(PSNR_All),np.mean(SSIM_All)
        """
def select_Q(A):
        #A = torch.from_numpy(A)
        A = torch.sum(A,dim=0)
        A[torch.eq(A, 0)] = 1
        Q = torch.unsqueeze(A,dim=0)
        Q = Q.expand([8,-1,-1])
        Q = 1/Q
        return Q    
    


if __name__=="__main__":

    results_saving_path = "results112" ##113
    model_name = "Tensor_AMP_Net_model"
    PhaseNumber = 10
    A = load_sampling_matrix()
    A = torch.from_numpy(A)
    #print(A)
    #A = torch.transpose(A,1,2)
    #print(A.shape)
    Q = select_Q(A)
    #print(A.shape)
    model = Tensor_AMP_net(PhaseNumber, A,Q)  # load the model
    path = os.path.join(results_saving_path,model_name,str(PhaseNumber),"best_model.pkl")
    model.cuda()
    model.load_state_dict(torch.load(path),False)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    p,s=get_val_result(model,PhaseNumber, is_cuda=True)  # test AMP_net
    print(p,s)

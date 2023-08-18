from dataset import *
#from nodeblock_model import Tensor_AMP_net
#from Unet import Tensor_AMP_net
from GAP_GRUacc import Tensor_AMP_net
import numpy as np
from torch.autograd import Variable
import math
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def load_sampling_matrix():
    path = "/home/hengling/lx/TensorAMP_Net/maskData/mask256.mat"
    mask = io.loadmat(path)["phi"]
    return mask
def select_Q(A):
        #A = torch.from_numpy(A)
        A = torch.sum(A,dim=0)
        A[torch.eq(A, 0)] = 1
        Q = torch.unsqueeze(A,dim=0)
        Q = Q.expand([8,-1,-1])
        Q = 1/Q
        return Q
def train(model, opt, train_loader, epoch, batch_size, PhaseNum):
    model.train()
    n = 0
    for data in train_loader:
        #print(data.shape)
        n = n + 1
        opt.zero_grad()  # 清空梯度
        # data = torch.unsqueeze(data,dim=1), torch.unsqueeze(target,dim=1)
        data = Variable(data.float().cuda())

        outputs= model(data, PhaseNum)        
        loss =0.5*torch.mean((outputs - data) ** 2)
        loss.backward()#23.36
        opt.step()
        #loss = model.for_backward(data,PhaseNum,opt)
        if n % 5 == 0:
            output = "PhaseNum: %d [%02d/%02d] loss: %.8f" % (
            PhaseNum, epoch, batch_size * n, loss.data.item())
            # output = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch, batch_size*n,
            #                                        cost.data.item(),cost_sym.data.item())
            print(output)
            break

def get_val_result(model, num, is_cuda=True):
    model.eval()
    test_set_path = "val.pt"
    val_dataset = dataset(root="/home/hengling/lx/TensorAMP_Net/training_data/NBA",train=False, transform=None,
                          target_transform=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             shuffle=False, num_workers=4)
    ImgNum = len(val_loader)  # 测试图像的数量
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        n = 0
        for data in val_loader:
            data = Variable(data.float().cuda())
            outputs= model(data, num)

            if is_cuda:
                outputs = outputs.cpu().data.numpy()
            else:
                outputs = outputs.data.numpy()
            rec_PSNR = psnr(outputs*255, data.cpu().data.numpy()*255)  # 计算PSNR的值
            PSNR_All[0, n] = rec_PSNR
            n+=1 
        out = np.mean(PSNR_All)
    return out



if __name__=="__main__":
    is_cuda = True
    #CS_ratio = 25  # 4, 10, 25, 30, 40, 50
    # n_output = 1089
    PhaseNumbers = [10]  # block 数目为 5
    # nrtrain = 88912
    learning_rate = 0.0001#25.6
    EpochNum = 500
    batch_size = 2
    results_saving_path = "results_meiyong"
    model_name = "Tensor_AMP_Net_model"
    #log_dir  = os.path.join("results112",model_name,str(7),"best_model.pkl")
    #print(log_dir)
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)
    results_saving_path = os.path.join(results_saving_path,model_name)
 
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    print('Load Data...')  # jiazaishuju

    A = load_sampling_matrix( )
    A = torch.from_numpy(A).cuda()
    Q = select_Q(A)
    #Q = torch.from_numpy(Q)
    train_dataset = dataset(root="/home/hengling/lx/TensorAMP_Net/training_data/all",train=True, transform=None,
                            target_transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    aaa = len(train_loader)
    for PhaseNumber in PhaseNumbers:
        model = Tensor_AMP_net(PhaseNumber,A,Q)  # load the model
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.cuda()
        #print(model)
        sub_path = os.path.join(results_saving_path, str(PhaseNumber))
        #model.load_state_dict(torch.load(log_dir),False)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        best_psnr = 0
        for epoch in range(1, EpochNum + 1):
            #if epoch == 50:
                #opt.defaults['lr'] *= 0.1
            if epoch == 300:
                opt.defaults['lr'] *= 0.1 
            if epoch == 400:
                opt.defaults['lr'] *= 0.1 
            train(model, opt, train_loader, epoch, batch_size, PhaseNumber)
            one_psnr = get_val_result(model, PhaseNumber)

            print_str = "Phase: %d epoch: %d  psnr: %.4f" % (PhaseNumber, epoch, one_psnr)
            print(print_str)

            output_file = open(sub_path + "/log_PSNR.txt", 'a')
            output_file.write("PSNR: %.4f\n" % (one_psnr))
            output_file.close()

            if one_psnr > best_psnr:
                best_psnr = one_psnr
                output_file = open(sub_path + "/log_PSNR_best.txt", 'a')
                output_file.write("PSNR: %.4f\n" % (best_psnr))
                output_file.close()
                torch.save(model.state_dict(), sub_path + "/best_model.pkl")
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

from Net.CNN import CNN


def train(EPOCH):
    save_path = r'./model/model.pt'
    batch_size = 64
    device = torch.device('cuda')
    net = CNN().to(device)
    # 导入文件并创建对应的dataLoader
    train_files = datasets.MNIST(root=r'./dataset', transform=transforms.ToTensor(), download=True, train=True)
    test_files = datasets.MNIST(root=r'./dataset', train=False, transform=transforms.ToTensor(), download=False)
    train_data = DataLoader(dataset=train_files, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(dataset=test_files, batch_size=batch_size, shuffle=False)
    # 设置网络的优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    # 两个list，分别用于存放训练过程中的损失和准确度
    loss_list = []
    acc_list = []
    # 开始训练
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (data, target) in enumerate(train_data, 1):
            # 优化器初始置零
            optimizer.zero_grad()
            # 将数据放入device
            data = data.to(device)
            target = target.to(device)
            # 获得当前网络的输出
            output = net(data)
            # 用输出和真实值计算损失
            loss = loss_func(output, target)
            # 反馈
            loss.backward()
            # 优化
            optimizer.step()
            # 记录本次的损失
            loss_list.append(loss.item())
            # 记录100次的损失
            running_loss += loss.item()
            if step % 100 == 0:
                print("Epoch:%d,第%d回:avg_loss=%.3f" % (epoch+1, step+1, running_loss/100))
                running_loss = 0.0
            # 获取当前的预测值
            _, predict = torch.max(output.data, 1)
            # 计算预测成功的个数
            correct = (predict == target).sum().item()
            total = target.size(0)
            # 计算并记录当前的准确度
            acc_list.append(100*correct/total)
    '''保存模型'''
    torch.save(net.state.dict(), save_path)
    '''训练结束，开始绘图'''
    train_iters = [x for x in range(1, len(acc_list)+1)]
    plt.title('training_loss')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.plot(train_iters, loss_list, color='red', label='train_cost')
    plt.grid()
    # 保存图片到本地
    plt.savefig(fname='result_loss.png', dpi=320, papertype='a4')
    # 清空
    plt.cla()
    plt.title("training_acc")
    plt.xlabel('iter')
    plt.ylabel('acc')
    plt.plot(train_iters, acc_list, color='blue', label='train_acc')
    plt.grid()
    plt.savefig(fname="result_acc.png", dpi=320, papertype='a4')


if __name__ == '__main__':
    train(10)

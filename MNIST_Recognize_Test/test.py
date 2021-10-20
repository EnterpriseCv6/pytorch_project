import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Net.CNN import CNN


def test():
    device = torch.device('cuda')
    net = CNN().to(device)
    batch_size = 64
    # 读取模型
    model_path = r'./model/model.pt'
    net.load_state_dict(torch.load(model_path))
    '''train_files = datasets.MNIST(root=r'./dataset', transform=transforms.ToTensor(), download=True, train=True)
    train_data = DataLoader(dataset=train_files, batch_size=batch_size, shuffle=True)'''
    test_files = datasets.MNIST(root=r'./dataset', train=False, transform=transforms.ToTensor(), download=False)
    test_data = DataLoader(dataset=test_files, batch_size=batch_size, shuffle=False)
    # 用与记录总的次数和成功次数
    total = 0
    correct = 0
    # 用于记录每一类（0-9）的次数和成功次数
    total_list = list(i for i in range(10))
    correct_list = list(i for i in range(10))

    # 因为是测试，所以梯度置零
    with torch.no_grad():
        for data in test_data:
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = net(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            # 遍历label的值，依次计算出现的次数和该类的预测成功次数
            for i in range(label.size(0)):
                total_list[label[i]] += 1
                if predicted[i] == label[i]:
                    correct_list[label[i]] += 1
    print('Accuracy:%.3f' % (correct / total))
    for i in range(10):
        print("Accuracy of %d : %.3f" % (i, correct_list[i] / total_list[i]))


if __name__ == "__main__":
    test()



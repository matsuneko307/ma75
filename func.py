import datetime
import glob
import yfinance as yf
import mplfinance as mpf
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import torch
import imagehash
import streamlit as st


def get_stock(code):
    """
    75日線直下の銘柄のデータを取得する関数
    """
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365)
    df = yf.download(f"{code}.T", start=start, end=today)

    return df


def under_75(df):
    """
    75日線直下かどうかを判定する関数
    """
    ma75 = df["Close"].rolling(window=75).mean()

    diff = (df["Close"].iloc[-1] / ma75.iloc[-1]) - 1
    if diff < -0.05 or diff > 0:
        return False
    else:
        return True
    

def get_graph(df):
    """
    75日線直下の銘柄のグラフを作成する関数
    """
    ma5 = df["Close"].rolling(window=5).mean()
    ma25 = df["Close"].rolling(window=25).mean()
    ma75 = df["Close"].rolling(window=75).mean()

    df = df.iloc[-60:]

    addplot = [
    mpf.make_addplot(ma5.iloc[-60:], color='red', width=1),
    mpf.make_addplot(ma25.iloc[-60:], color='blue', width=1),
    mpf.make_addplot(ma75.iloc[-60:], color='green', width=1)
    ]

    s = mpf.make_mpf_style(gridstyle=' ', y_on_right=False)

    mpf.plot(df, type='line', style=s, linecolor='white', axisoff=True, addplot=addplot, savefig='graph.png')

    img = Image.open('graph.png')

    return img


#imgを加工する関数
def trans_img(img):

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if np.array(img).shape[2] == 4:
        img = img.convert("RGB")
        img = transform(img)
        image_tensor = img.unsqueeze(0)  # バッチ次元を追加
    else:
        img = transform(img)
        image_tensor = img.unsqueeze(0)  # バッチ次元を追加

    return image_tensor


#予測する関数
def predict(image_tensor, path="model.pth"):

    class Net(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.num_classes = num_classes
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

        def forward(self, x):
            return self.resnet(x)

    # 推論
    net = Net().cpu().eval()
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    with torch.no_grad():
        output = net(image_tensor)
        predicted_class = torch.argmax(output, dim=1)
    predict = predicted_class.item()
    
    return predict


def imghash(img, predict):

    dic = {}
    code_list = []
    date_list = []
    diff_list = []

    paths = glob.glob(f"{predict}/*.png")

    hash = imagehash.average_hash(img)
    for path in paths:
        hash = imagehash.average_hash(img)
        hash2 = imagehash.average_hash(Image.open(path))
        dic[path] = hash - hash2
        #dicをvalueでソート
    dic = sorted(dic.items(), key=lambda x:x[1])

    for i in range(15):
        code_list.append(dic[i][0].split("\\")[1].split("_")[0])
        date_list.append(dic[i][0].split("\\")[1].split("_")[1])
        diff_list.append(dic[i][1])

    return dic, code_list, date_list, diff_list
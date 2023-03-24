import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
from torchvision import transforms
from PIL import Image
import glob
import torch.nn as nn
from torchvision.models import resnet18
import imagehash
import mplfinance as mpf
import func


st.set_page_config(layout="wide")
st.title('過去の似た動きをした銘柄を探すアプリ')
st.write('75日線直下(乖離率-5%以上0%以下)にある銘柄が、どの銘柄の値動きと似ているかを5，25，75日線を使って判定します。')
st.write("------------------------------------------------------------------------------------------------------------------------------------")
st.write('使い方')
st.write('左側に銘柄コード入力後、分析開始ボタンを押してください。75日線直下ではない場合、分析できません。')
st.write('ボタンを押すと、アップロードしたグラフと似た動きをした過去の銘柄を探します。')

#グラフを作成する
st.sidebar.write('分析したい会社の証券コードを入力してください')
code = st.sidebar.text_input('証券コード')

if st.sidebar.button('分析開始') and code:
    df = func.get_stock(code)
    
    if func.under_75(df):
        img = func.get_graph(df)

        st.header(f'{code}')
        st.image(img, use_column_width=True)
        st.write(f'{code}の移動平均線グラフを作成しました。')
        st.write('似た動きをした銘柄を探します。類似差が小さいほど似ています。')
    else:
        st.write('75日線直下ではありません。75日線乖離率が-5%以上0%未満の証券コードを入力してください。')
        st.stop()

    #画像を読み込む
    img = Image.open('graph.png')
    #imgをtensorに変換
    img_tensor = func.trans_img(img)
    
    predict = func.predict(img_tensor)
    
    dic, code_list, date_list, diff_list = func.imghash(img, predict)

    num = len(code_list)

    st.sidebar.table(pd.DataFrame({'コード':code_list, '日付':date_list, '類似差':diff_list}))
    for i in range(0,num):
            st.header(f"コード：{code_list[i]}, 日付：{date_list[i]}, 類似差：{diff_list[i]}")
            st.image(dic[i][0])

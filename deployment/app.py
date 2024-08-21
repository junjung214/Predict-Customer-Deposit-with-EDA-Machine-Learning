import streamlit as st
import pandas as pd
import joblib

st.header('Milestone 2 Model Deployment')
st.write("""
Created By Ahmad Junjung S - RMT 033

Dalam project ini akan membuat suatu model machine learning yang dimana bisa memprediksi orang orang atau konsumen terhadap subscibe deposit di suatu bank. Sehingga dalam pengembangan project ini maka diperlukan beberapa tahapan yang bisa menunjang untuk keberhasilan seperti EDA, Preprocessing, Modeling dan Evaluasi terhadap model. Dengan adanya project ini diharapkan dapat mempermudah suatu bank untuk memprediksi orang orang yang akan berkecenderungan memilih untuk subcrobe berdasarkan model yang dibuat
""")

#load data
df = pd.read_csv('bank-additional-full.csv', sep=';')
st.write(df)

#buat sidebar untuk inputan user
st.sidebar.header('User Input Features')

def user_input():
    age = st.sidebar.slider('Silahkan isi Umur Anda : ', 0 , 150, 30)
    job = st.sidebar.selectbox('Silahkan isi Perkerjaan Anda :', df['job'].unique())
    martial = st.sidebar.selectbox('Silahkan isi Status Kawin anda :', df['marital'].unique())
    education = st.sidebar.selectbox('Silahkan Isi Status Pendidikan Anda : ', df['education'].unique())
    default = st.sidebar.select_slider('Apakah anda memilki kartu kredot :', df['default'].unique())
    housing = st.sidebar.select_slider('Apakah anda memilki pinjaman rumah :', df['housing'].unique())
    loan = st.sidebar.select_slider('Apakah anda memiliki pinjaman pribadi : ', df['loan'].unique())
    contact = st.sidebar.select_slider('Jenis Teknologi Komunikasi yang digunakan :', df['contact'].unique())
    month = st.sidebar.selectbox('Bulan berapa kontak terakhir : ', df['month'].unique())
    day_of_week = st.sidebar.selectbox('Pada hari apa anda kontak :', df['day_of_week'].unique())
    duration = st.sidebar.number_input('Berapa lama anda melakukan kontak (detik) :', value=100)
    campaign = st.sidebar.number_input('Jumlah Campaign ketika ketika melakukan kontak :' , value=20)
    pdays = st.sidebar.number_input('Jumlah Hari berlalu setelah kontak dengan konsumen apabila belum pernah maka 999:', value=999)
    previous = st.sidebar.number_input('Jumlah Kampanya yang dilakukan sebelum kontak terakhir : ', value=0)
    poutcome = st.sidebar.selectbox('Hasil Status Kampanye : ', df['poutcome'].unique())
    emp_var_rate = st.sidebar.number_input('Tingkat Variasi Ketenagakerjaan - indikator kuartalan :', value = 1)
    cons_price_idx = st.sidebar.number_input('Indeks harga konsumen - indikator bulanan', value=0)
    cons_conft_idx = st.sidebar.number_input('Index Kepercayaan Konsumen : ', value=0)
    euribor3m = st.sidebar.number_input('Tingkat euribor 3 bulan - indikator harin : ', value=1)
    nr_employed = st.sidebar.number_input('Jumlah Karyawan - indikator bulanan :', value=0)

    data = {
            'duration' : duration,
            'campaign' : campaign,
            'emp.var.rate' : emp_var_rate,
            'euribor3m' : euribor3m,
            'job' : job,
            'marital' : martial,
            'education' : education,
            'default' : default,
            'housing' : housing,
            'loan' : loan,
            'contact' : contact,
            'month' : month,
            'day_of_week' : day_of_week,
            'poutcome' : poutcome
        }
    features = pd.DataFrame(data, index=[0])
    return features

input = user_input()
st.write(input)
#load data
pipe_cb = joblib.load('pipe_cb.pkl')
#data final
data_final = input.copy()
#predict
if st.button('predict'):
    prediciton = pipe_cb.predict(data_final)
    if prediciton == 'no':
        prediction = 'Tidak Subscribe'
    else : 
        prediction = 'Subscribe'

    st.write('Berdasarkan user input, model dapat memprediksi sebagai berikut :')
    st.write(prediction)
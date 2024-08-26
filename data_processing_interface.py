from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import category_encoders as ce

data = 'car_evaluation.csv'

df = pd.read_csv(data, header=None)

while True:

    user_input=input("1) Dataset'i incele ve sütunları isimlendir\n2) Null varmı yokmu incele\n3) Frekanslara bak\n4) Özellikleri encode et\n5) Korelasyona bak\nRegression'a devam etmek için başka birşey girin.\n")

    if user_input=="1":
        print("Datasetin şekli: ", df.shape)
        #Outputs (1728, 7)

        print("Datasetin head'i:\n", df.head())

        #Sütunlarımızın isimleri olmadığı için onları isimlendiriyoruz

        col_names = ['satis_fiyati', 'bakim_masrafi', 'kapi_sayisi', 'kisi_kapasitesi', 'bagaj_kapasitesi', 'guvenlik', 'sinif']

        df.columns = col_names
        print("Yeniden isimlendirmeden sonra datasetin head'i:\n", df.head())
    elif user_input=="2":
        print("Info of our dataset:\n", df.info())
        print("Null sayısı:\n", df.isnull().sum())
        #Datasetimizde hiç null alan yok
    elif user_input=="3":
        print("Sütunlardaki her değerin sayısı (frekansı):\n")
        for col in col_names:
            print(df[col].value_counts())
            #Frekanslara bakılırsa "sınıf" özelliğini target almamız en iyisi olur.
    #####  
        """
        Feature'larımızın hepsi projemizin context'i dolayısıyla kategorik veriden oluştuğu için ve objelerin testlerimize göre int64 olarak 
        göründüğü için, onları bir encoding methoduyla kategorik verilere çevirmemiz gerek. "category_encoders" kütüphanesi içinde birçok farklı
        encoding methodu bulundurur ve bunlar yüksek kardinalitesi olan (feature'un çok sayıda kategorisi olan) datasetler için label encoding ve
        one-hot-encoding'e kıyasla daha uygun methodlar bulundurur. Bu yüzden category_encoding'i kullanmamız daha uygun olabilir.
        """
    elif user_input=="4":
        encoder = ce.OrdinalEncoder(cols=['satis_fiyati', 'bakim_masrafi', 'kapi_sayisi', 'kisi_kapasitesi', 'bagaj_kapasitesi', 'guvenlik', 'sinif'])
        df=encoder.fit_transform(df)
        print("Encoding sonrası datasetin head'i:\n", df.head())
    elif user_input=="5":
        plt.figure(figsize=(18,10))
        heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        plt.show()

        print(df.corr())

        # En yüksek mutlak 5 korelasyonu sıralayalım
        for col in df:
            en_yuksek_degerler = abs(df.corr()[col]).nlargest(n=5)  
            # en yüksek korelasyona sahip 5 değeri alalım
            print(en_yuksek_degerler)
            # eğer 0.75'ten büyük değer varsa yazdır.
            for index, value in en_yuksek_degerler.items():
                if 1 > value >= 0.75:
                    print(index, col, "değişkenleri yüksek korelasyona sahip: ", value)
    else:
        break
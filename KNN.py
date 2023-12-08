#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Outcome = 1 Diabet/Şeker Hastası
#Outcome = 0 Sağlıklı
data = pd.read_csv("diabetes.csv")
data.head()


# In[9]:


seker_hastalari = data[data.Outcome==1]
saglikli_insanlar = data[data.Outcome==0]

#Şimdilik sadece gloucose'a bakarak örnek bir çizim yapalım.
#Programımızın sonunda makine öğrenme modelimiz sadece glikoza değil tüm diğer verilere bakarak bir tahmin yapacaktır.

plt.scatter(saglikli_insanlar.Age,saglikli_insanlar.Glucose,color="green",label="Sağlıklı",alpha = 0.4)
plt.scatter(seker_hastalari.Age,seker_hastalari.Glucose,color="red",label="Diabet Hastası",alpha = 0.4)

plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


# In[10]:


#x ve y eksenlerini belirleyelim.
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)
#Outcome sütununu (dependet variable) çıkarıp geriye kalanını (Outcome hariç) independet variables bırakıyoruz...
#Çünkü KNN algoritması x değerleri içerisinde gruplandırma yapacaktır...

#Normalization yapıyoruz -x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz..
#Eğer bu şekilde normalization yapmazsak yüksek rakamlar küçük rakamları ezer ve KNN algoritmasını yanıltabilir!
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri) - np.min(x_ham_veri))
#(bütün ham veri değerleri- ham veri minleri) / (ham veri maxları - ham veri minleri) HER BİRİ İÇİN NORMALİZE EDECEK.

#Önce
print("Normalization öncesi ham veriler:  \n ")
print(x_ham_veri.head())

#Sonra
print("\n \n \n Normalization sonrası yapay zekaya eğitim için vereceğimiz veriler: \n")
print(x.head())


# In[11]:


#Train datamız ile test datamızı ayırıyoruz.
#Train datamız sistemin sağlıklı insan ile hasta insanı ayırt etmesini öğrenmek için kullanılacak.
#Test datamız ise bakalım makine öğrenme modelimiz doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye
#test etmek için kullanılıcak...
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)

#KNN Modelimizi oluşturuyoruz.
knn = KNeighborsClassifier(n_neighbors = 3) #n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için Test verilerimizin doğrulama testi sonucu",knn.score(x_test,y_test))

# k kaç olmalı?
#en iyi k değerini belirleyelim...
sayac = 1 
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac," ","Doğruluk oranı: %",knn_yeni.score(x_test,y_test)*100)
    sayac += 1






# In[12]:


# Yeni bir hasta tahmini için:
from sklearn.preprocessing import MinMaxScaler

# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]


# In[ ]:





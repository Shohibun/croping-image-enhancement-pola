import cv2 #bertujuan untuk memanggil modul library cv2 yaitu OpenCV (Open Source Computer Vision Library)
import glob #bertujuan untuk membaca direktori folder sekarang 
import numpy as np #bertujuan menambahkan dukungan untuk array dan matriks multidimensi
import matplotlib.pyplot as plt #bertujuan untuk membuat visualisasi data yang statis, animasi, dan interaktif
import pandas as pd #bertujuan untuk membuat tabel, mengubah dimensi data, mengecek data
import arff #format weka arff
import seaborn as sns #bertujuan untuk membuat grafik dan statistik
from os import listdir
from os.path import isfile, join
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix


def Menu():
    print("=====Daftar Menu Pengolahan Citra=====")
    print("[1] Crop")
    print("[2] Preprocessing")
    print("[3] Segmentasi")
    print("[4] Ekstraksi Fitur")
    print("[5] Klasifikasi")
    print("[6] Exit")

    pilihan = input("Pilih program yang ingin anda jalankan: ")
    if pilihan.lower() == "1":
        Crop()
    elif pilihan.lower() == "2":
        Preprocessing()
    elif pilihan.lower() == "3":
        Segmentasi()
    elif pilihan.lower() == "4":
        EkstraksiFitur()
    elif pilihan.lower() == "5":
        Klasifikasi()
    elif pilihan.lower() == "6":
        exit()
    else:
        print("Program tidak ada di menu")

def BackToMenu():
    Menu()

def Crop():
    def resizeImage(imageFile, indexKey): #fungsi yang menampung source code untuk mengubah resolusi ukuran citra 
        image = cv2.imread(imageFile) #cv2.imread berguna untuk memanggil citra yang akan diolah

        r = 100.0 / image.shape[1] #r adalah lebar citra
        dim = (100, int(image.shape[0] *r)) #dim adalah panjang citra

        imageResized = cv2.resize(image, (1000, 750), dim, interpolation = cv2.INTER_AREA) #cv2.resize berguna utnuk mengubah ukuran citra 

        cv2.imwrite('Crop/imageresized_{}.png'.format(indexKey), imageResized) #cv2.imwrite berguna untuk menulis hasil citra kedalam suatu file yang dituju 
        print('imageresized_{}.png'.format(indexKey))

    #iglob bertujuan untuk membaca semua file yang berformat .jpg 
    #x = index 
    #imageFile adalah file dataset 
    #enumerate ngelooping file yang berformat jpg (list)
    #[(0, "Satu.jpg"), (1, "Dua.jpg)]
    for (x, imageFile) in enumerate(glob.iglob('*.jpg')): 
        resizeImage(imageFile, x)
    
    BackToMenu()

def Preprocessing():
    def brightening(imageFile, indexKey): #fungsi yang menampung proses pengolahan brightening 
        img = cv2.imread(imageFile) #cv2.imread berguna untuk memanggil citra yang akan diolah
        dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) #digunakan untuk melakukan denoising (menghilangkan noise) terhadap citra 
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #sebagai paramaeter simg
        simg=cv2.filter2D(dst,-1,filter)  #untuk lebih memperjelas objek menggunakan metode shappening 
        
        contrast = 1.0 #kontras
        brightness = 45 #pencerahan
        imgbaru = np.zeros(simg.shape, simg.dtype) 
        
        #memanipulasi RGB
        for y in range(simg.shape[0]):
            for x in range(simg.shape[1]):
                for c in range(simg.shape[2]):
                    imgbaru[y,x,c] = np.clip(contrast*simg[y,x,c] + brightness, 0, 255)
        
        cv2.imwrite('Brightening_{}.png'.format(indexKey), imgbaru) #cv2.imwrite berguna untuk menulis hasil citra kedalam suatu file yang dituju 
        print('Brightening_{}.png'.format(indexKey))

    #iglob bertujuan untuk membaca semua file yang berformat .png 
    #x = index 
    #imageFile adalah file dataset 
    #enumerate ngelooping file yang berformat jpg (list)
    #[(0, "Satu.jpg"), (1, "Dua.jpg)]
    for (x, imageFile) in enumerate(glob.iglob('*.png')):
        brightening(imageFile, x)
    
    BackToMenu()

def Segmentasi():
    def masked_image(image, mask):
        r = image[:,:,0] * mask
        g = image[:,:,1] * mask
        b = image[:,:,2] * mask
        return np.dstack([r,g,b])
    
    def detection(imageFile, indexKey):
        img = cv2.imread(imageFile)
        th_values = np.linspace(0, 1, 11)
        fig, axis = plt.subplots(2, 5, figsize=(15,8))
        chico_gray = rgb2gray(img)

        for th, ax in zip(th_values, axis.flatten()):
            chico_binarized = chico_gray < th
            ax.imshow(chico_binarized,cmap='binary')
            ax.set_title('$Threshold = %.2f$' % th)

        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        thresh = threshold_otsu(chico_gray)
        chico_otsu  = chico_gray < thresh
        filtered = masked_image(img, chico_otsu)

        cv2.imwrite('Detection_{}.jpg'.format(indexKey), filtered)
        print('Detection_{}.jpg'.format(indexKey))   

    #iglob bertujuan untuk membaca semua file yang berformat .jpg 
    #x = index 
    #imageFile adalah file dataset 
    #enumerate ngelooping file yang berformat jpg (list)
    #[(0, "Satu.jpg"), (1, "Dua.jpg)]
    for (x, imageFile) in enumerate(glob.iglob('*.png')):
        detection(imageFile, x) 
    
    BackToMenu()

def EkstraksiFitur():
    def getRata2BGR(img_path):
        img = cv2.imread(img_path)
        h,w,ch = img.shape[:]
        imgCopy = img.copy()
        
        #nilai threshold
        bbr, bbg, bbb = 0,0,0  #batas bawah
        bar, bag, bab = 200,200,200 #batas atas
        
        batas_bawah = np.array((bbr, bbg, bbb), np.uint8)
        batas_atas = np.array((bar, bag, bab), np.uint8)
        img_threshold = cv2.inRange(imgCopy, batas_bawah, batas_atas)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHasil = cv2.bitwise_and(imgCopy, hsv, img_threshold)
        
        avg_color_per_row = np.average(imgHasil, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        jml_nonzero = cv2.countNonZero(img_threshold)
        
        avg_color = avg_color * (h*w)/jml_nonzero
        
        #cek image
        cv2.imshow('bgrGry', hsv)
        cv2.imshow('imgHasil', imgHasil)
        cv2.imshow('img', img)
        return avg_color
    
    #list directory
    arr_dir = ['Castor Aralia','Japanese Maple']
    fnames = []
    for a in range (len(arr_dir)):
        #get all files name
        arr_tmp = [f for f in listdir(arr_dir[a]) if isfile (join(arr_dir[a],f))]

        fnames.append(arr_tmp)

    df = pd.DataFrame(columns=['b', 'g', 'r', 'class'])
    baris = 0
    for i in range(len(arr_dir)):
        for j in range(len(fnames[i])):
            print(arr_dir[i]+'/'+fnames[i][j])
            avg = getRata2BGR(arr_dir[i]+'/'+fnames[i][j])
            print(avg)
            ab,ag,ar = avg
            rows_list = [ab,ag,ar,arr_dir[i]]
            df.loc[baris] = rows_list
            baris = baris + 1
            
    cv2.waitKey(20)
    print(df) 
    #simpan format csv
    df.to_csv ('data.csv', index = False, header=True)
    #simpan format weka arff
    arff.dump('data.arff'
            , df.values
            , relation='Ekstraksi Warna Pada Daun Castor Aralia dan Japanese Maple'
            , names=df.columns)

    plt.figure()
    sns.scatterplot(data = df
                    ,x = 'b'
                    ,y = 'g'
                    ,hue = 'class'
                    )

    plt.figure()
    sns.scatterplot(data = df
                    ,x = 'b'
                    ,y = 'r'
                    ,hue = 'class'
                    )

    plt.figure()
    sns.scatterplot(data = df
                    ,x = 'g'
                    ,y = 'r'
                    ,hue = 'class'
                    )
    plt.show()

    BackToMenu()

def Klasifikasi():
    #read data 
    df = pd.read_csv('data.csv') #data hasil ekstraksi fitur 
    #check data has been read in properly
    df.head()

    print(df)

    #create a dataframe with all training data except the target column
    X = df.drop(columns=['class'])
    #check that the target variable has been removed
    X.head()
    print('X:')
    print(X)

    #separate target values
    y = df['class'].values

    print('y:')
    print(y)

    from sklearn.model_selection import train_test_split
    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y) 

    print('X_train: ')
    print(X_train)
    print('X_test: ')
    print(X_test)
    print('y_train: ')
    print(y_train)
    print('y_test: ')
    print(y_test)

    from sklearn.neighbors import KNeighborsClassifier
    #Create KNN classifier
    jml_k = 3
    knn = KNeighborsClassifier(n_neighbors = jml_k)
    #Fit the classifier to the data
    knn.fit(X_train,y_train)

    #show first 5 model predictions on the test data
    result = knn.predict(X_test)
    print('==============================')

    print('y_test (true class): \n', y_test)

    print('result (predicted class):\n', result)

    akurasi = knn.score(X_test, y_test)
    print('\nakurasi: %.3f%%' % (100*akurasi)) 
    

    conf_matrix = confusion_matrix(y_test, result )
    print('================================ ')
    print('Confusion matrik data testing: ')
    print(conf_matrix)


    plt.clf()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix Testing Data, Daun Castor Aralia dan Japanese Maple')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    #~ s = [['TN (True negatif)','FP (False positif)'], ['FN (False negatif)', 'TP (True Positif)']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(conf_matrix[i][j]))
    plt.show()

    BackToMenu()

def exit():
    exit

if __name__ == "__main__":
    Menu()


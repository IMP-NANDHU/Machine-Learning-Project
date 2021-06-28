from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
import tkinter
from tkinter import *
from tkinter import filedialog
import pandas as pd
from tkinter import ttk
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


class pages:

    def window1(self):
        window1 = Tk()

        label1 = Label(window1,text="ML Algorithms Software", bg="#ffffff",fg='#000000', font=("Bookman Old Style",20,'bold'))
        label1.pack(pady=20)

        label2 = Label(window1, text="Upload your Dataset", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
        label2.pack(pady=20)

        def browse_data():
            global input, data
            input = filedialog.askopenfilename(initialdir="/",filetypes = [("csv files","*.csv")])
            data = pd.read_csv(input)
            entry1.insert(0, input)

        entry1 = Entry(window1,bd=3)
        entry1.place(x=130, y=200, height=30, width=380)

        button1 = Button(window1, text='Browse',bg="#ffb700", fg="#000000", command=lambda: browse_data())
        button1.place(x=500, y=200, height=30, width=80)

        button2 = Button(window1, text='Next', bg="#03cdff",command=p.window2)
        button2.place(x=300, y=250, height=30, width=70)

        window1.configure(bg="#ffffff")
        window1.geometry("700x500+100+10")
        window1.title("Upload your dataset")
        window1.mainloop()


    def window2(self):
        window2 = Tk()

        label3 = Label(window2, text="Confirm your dataset", bg="#ffffff", fg='#000000', font=("Bookman Old Style", 20, 'bold'))
        label3.pack(pady=20)

        frame = Frame(window2)
        frame.pack(padx=20)

        scroll1 = ttk.Scrollbar(frame, orient='vertical')
        scroll1.pack(side=RIGHT, fill=Y)

        scroll2 = ttk.Scrollbar(frame, orient='horizontal')
        scroll2.pack(side=BOTTOM, fill=X)

        global tview
        tview = ttk.Treeview(frame, yscrollcommand=scroll1.set, xscrollcommand=scroll2.set)
        tview.pack()

        scroll1.config(command=tview.yview)
        scroll2.config(command=tview.xview)

        tview["column"] = list(data.columns)
        tview["show"] = "headings"

        for column in tview["column"]:
            tview.heading(column, text=column)

        # Put data in tree view
        df_rows = data.to_numpy().tolist()
        for row in df_rows:
            tview.insert("", "end", values=row)
        # pack the tree view finally
        tview.pack()


        confirm = PhotoImage(master=window2,file='confirm1.png')
        notconfirm = PhotoImage(master=window2, file='notconfirm1.png')


        button3 = Button(window2, image=confirm, borderwidth=0,bg="#ffffff",command=p.window3)
        button3.place(x=280,y=350)

        button4 = Button(window2, image=notconfirm, borderwidth=0, bg="#ffffff", command=p.window1)
        button4.place(x=370,y=350,width=90)

        window2.configure(bg="#ffffff")
        window2.title("DATASET_VIEW")
        window2.geometry("700x500+100+10")
        window2.mainloop()

    def window3(self):
        window3 = Tk()

        label31 = Label(window3, text="Select the Features", fg='#000000',bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
        label31.pack()

        label32 = Label(window3, text="Independent Variable", fg='#000000', bg="#ffffff" ,font=("Bookman Old Style", 15, 'bold'))
        label32.pack(padx=63,side=LEFT,anchor="n")

        label33 = Label(window3, text="Dependent Variable", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 15, 'bold'))
        label33.pack(padx=75,side=LEFT,anchor="n")

        frame1 = Frame(window3)
        frame1.place(x=65,y=75,width=220,height=150)

        scrollbar1 = Scrollbar(frame1)
        scrollbar1.pack(side=RIGHT,fill=BOTH )


        listbox1 = Listbox(frame1, selectmode=MULTIPLE)
        listbox1.config(yscrollcommand=scrollbar1.set)
        listbox1.place(width=205, height=150)

        scrollbar1.config(command=listbox1.yview)

        j=0
        for i in tview["column"]:
            listbox1.insert(j, i)
            j = j + 1

        frame2 = Frame(window3)
        frame2.place(x=420, y=75, width=220, height=150)

        scrollbar2 = Scrollbar(frame2)
        scrollbar2.pack(side=RIGHT, fill=BOTH)


        listbox2 = Listbox(frame2, selectmode=SINGLE)
        listbox2.config(yscrollcommand=scrollbar2.set)
        listbox2.place(width=205, height=150)

        scrollbar2.config(command=listbox2.yview)

        j = 0
        for i in tview["column"]:
            listbox2.insert(j, i)
            j = j + 1


        def indep_select():
            global indep, indep1
            indep = []
            indep1 = []
            label34 = Label(window3, text="Independent Variable :",bg="#ffffff")
            label34.place(x=65, y=260)
            clicked = listbox1.curselection()
            z = 280
            for item in clicked:
                label10 = Label(window3, text=listbox1.get(item),bg="#ffffff")
                label10.place(x=65, y=z)
                indep.append(item)
                indep1.append(listbox1.get(item))
                z = z + 15


        button31 = Button(window3,text="Select",command=indep_select)
        button31.place(x=140,y=225,width=60, height=30)

        def dep_select():
            global dep,dep1
            dep = []
            dep1 = []
            label34 = Label(window3, text="Dependent Variable :",bg="#ffffff")
            label34.place(x=420, y=260)
            clicked = listbox2.curselection()
            z = 280
            for item in clicked:
                label10 = Label(window3, text=listbox2.get(item),bg="#ffffff")
                label10.place(x=420, y=z)
                dep.append(item)
                dep1.append(listbox2.get(item))
                z = z + 15


        button32 = Button(window3, text="Select", command=dep_select)
        button32.place(x=495, y=225, width=60, height=30)

        button33 = Button(window3, text = "Confirm", command=p.window4)
        button33.place(x=320, y=350, height=30, width=80)

        label35 = Label(window3,text = "NOTE: If you are going to do Clustering, You have to select Independent Variables Only!",fg='#ff0000', bg="#ffffff", font=("Bookman Old Style", 10,'bold','underline'))
        label35.place(x=50,y=450)

        window3.configure(bg="#ffffff")
        window3.title("Feature Selection")
        window3.geometry("700x500+100+10")
        window3.mainloop()

    def window4(self):
        window4 = Tk()

        label31 = Label(window4, text="Select the Problem Type", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
        label31.place(x=200,y=180,height=35)

        button31 = Button(window4, text="Regression", bg="#ffb700", fg="#ffffff",command=p.regression)
        button31.place(x=90,y=250,width=130,height=40)

        button31 = Button(window4, text="Classification", bg="#ff0095", fg="#ffffff",command=p.classification)
        button31.place(x=290,y=250,width=130,height=40)

        button31 = Button(window4, text="Clustering", bg="#00d0ff", fg="#ffffff", command=p.clustering)
        button31.place(x=490,y=250,width=130,height=40)

        window4.configure(bg="#ffffff")
        window4.title("PROBLEM TYPE")
        window4.geometry("700x500+100+10")
        window4.mainloop()

    def regression(self):
        window5 = Tk()
        global algos
        algos="Regression"

        label3 = Label(window5, text="SELECT YOUR ALGORITHM", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20,"bold"))
        label3.pack(pady=10)

        def comboclick(event):
            X = data.iloc[:, indep].values  # Independent variable
            y = data.iloc[:, dep].values  # Dependent variable
            if (combo.get() == "Simple Linear Regression"):
                label = Label(window5, text="You have selected Simple Linear Regression!").pack()

                # Splitting the dataset into training and testing set.
                X_train, X_test, y_train, y_test = tts(X, y, test_size=1 / 3, random_state=0)

                # Fitting the simple linear regression model to the training dataset
                global regressor
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)

                global acc
                acc = regressor.score(X_train, y_train)


            elif (combo.get() == "Multiple Linear Regression"):
                label = Label(window5, text="You have selected Multiple Linear Regression!").pack()

                # Splitting the dataset into training and test set.
                x_train, x_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)

                regressor = LinearRegression()
                regressor.fit(x_train, y_train)

                acc = regressor.score(x_train, y_train)

            elif (combo.get() == "Polynomial Regression"):
                label = Label(window5, text="You have selected Polynomial Regression!").pack()

                # Training the dataset in polynomial regression
                from sklearn.preprocessing import PolynomialFeatures
                global poly_reg
                poly_reg = PolynomialFeatures(degree=5)  # Change the degree values to get the acuuracy
                x_poly = poly_reg.fit_transform(X)
                regressor = LinearRegression()
                regressor.fit(x_poly, y)

                acc = regressor.score(x_poly,y)

            elif(combo.get()=="SVM Regression"):
                label = Label(window5, text="You have selected SVM Regression!").pack()

                #Standardizarion
                global sc_X,sc_y
                sc_X = StandardScaler()
                sc_y = StandardScaler()
                X = sc_X.fit_transform(X)
                y = sc_y.fit_transform(y)

                regressor = SVR(kernel='rbf')
                regressor.fit(X,y)

                acc = regressor.score(X,y)

            elif(combo.get()=="Decision Tree Regression"):
                label = Label(window5, text="You have selected Decision Tree Regression!").pack()

                regressor = DecisionTreeRegressor(random_state=0)
                regressor.fit(X,y)

                acc = regressor.score(X,y)

            elif(combo.get()=="Random Forest Regression"):
                label = Label(window5, text="You have selected Decision Tree Regression!").pack()

                regressor = RandomForestRegressor(n_estimators=10,random_state=0)
                regressor.fit(X,y)

                acc=regressor.score(X,y)

        options = ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression","SVM Regression","Decision Tree Regression","Random Forest Regression"]
        global combo
        combo = ttk.Combobox(window5, value=options)
        combo.config(width=50)
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", comboclick)
        combo.pack()

        button6 = Button(window5, text='Train Model', bg="#ff0095", fg="#ffffff", command=p.window6)
        button6.pack(pady=20)

        window5.configure(bg="#ffffff")
        window5.title("Regression")
        window5.geometry("700x500+100+10")
        window5.mainloop()

    def classification(self):
        window8 = Tk()

        global algos
        algos="Classification"

        label3 = Label(window8, text="SELECT YOUR ALGORITHM", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20,"bold"))
        label3.pack(pady=10)

        def comboclick1(event):
            X = data.iloc[:, indep].values
            y = data.iloc[:, dep].values

            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)

            global sc
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            if(combo2.get()=="Logistic Regression"):
                label = Label(window8, text="NOTE: In Logistic Regression '0' means FALSE '1' means TRUE",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                global classifier
                classifier = LogisticRegression(random_state=0)
                classifier.fit(X_train,y_train)

                y_pred = classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                global acc
                acc = (sum(np.diag(cm))/len(y_test))

            elif(combo2.get()=="Naive Bayes Classification"):
                label = Label(window8, text="NOTE: If there is only two possible outcomes then '0' means FALSE '1' means TRUE!",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                classifier = GaussianNB()
                classifier.fit(X_train ,y_train)

                y_pred = classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = (sum(np.diag(cm))/len(y_test))

            elif(combo2.get()=="K-Nearest Neighbour Classification"):
                label = Label(window8, text="NOTE: If there is only two possible outcomes then '0' means FALSE '1' means TRUE!",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
                classifier.fit(X_train,y_train)

                y_pred = classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="SVM Classification"):
                label = Label(window8, text="NOTE: If there is only two possible outcomes then '0' means FALSE '1' means TRUE!",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                classifier = SVC(kernel='rbf',random_state=0)
                classifier.fit(X_train,y_train)

                y_pred=classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="Decision Tree Classification"):
                label = Label(window8, text="NOTE: If there is only two possible outcomes then '0' means FALSE '1' means TRUE!",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
                classifier.fit(X_train,y_train)

                y_pred = classifier.predict(X_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combo2.get()=="Random Forest Classification"):
                label = Label(window8, text="NOTE: If there is only two possible outcomes then '0' means FALSE '1' means TRUE!",fg='#ff0000', font=("Bookman Old Style", 10)).pack()

                classifier = RandomForestClassifier()
                classifier.fit(X_train,y_train)

                y_pred = classifier.predict(X_test)
                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

        options1 = ["Logistic Regression","Naive Bayes Classification","K-Nearest Neighbour Classification","SVM Classification","Decision Tree Classification","Random Forest Classification"]
        global combo2
        combo2 = ttk.Combobox(window8, value=options1)
        combo2.config(width=50)
        combo2.current(0)
        combo2.bind("<<ComboboxSelected>>", comboclick1)
        combo2.pack()

        button6 = Button(window8, text='Train Model', bg="#ff0095", fg="#ffffff",command=p.window6)
        button6.pack(pady=20)

        window8.configure(bg="#ffffff")
        window8.geometry("700x500+100+10")
        window8.title("Classification Algorithms")
        window8.mainloop()


    def clustering(self):
        window9 = Tk()
        global algos
        algos = "clustering"

        label31 = Label(window9, text="SELECT YOUR ALGORITHM", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20,"bold"))
        label31.pack(pady=10)

        def comboclick2(event):
            X = data.iloc[:, indep].values
            if(combo3.get()=="KMeans Clustering"):

                wcss=[]

                for i in range(1,11):
                    kmeans = KMeans(n_clusters=i,random_state=0)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
                def plot():
                    button92 = Button(window9, text="Next", command=lambda:[kmeanspredict(),p.window7()])
                    button92.place(x=320, y=220, width=50)

                    plt.figure(figsize = (8,5), dpi=50)
                    plt.scatter(range(1,11),wcss)
                    plt.plot(range(1,11),wcss)
                    plt.title("Elbow Method")
                    plt.xlabel("Number of Clusters")
                    plt.ylabel("WCSS")
                    plt.show()

                button91 = Button(window9, text="Click to view the plot", command=plot)
                button91.place(x=290,y=80)

                label32 = Label(window9,text="Enter the number of clusters :",fg='#fa00b7', bg="#ffffff", font=("Bookman Old Style", 10))
                label32.place(x=215,y=180)

                entry31 = Entry(window9)
                entry31.place(x=425,y=180,height=25,width=50)

                def kmeanspredict():
                    n = int(entry31.get())
                    kmeans = KMeans(n_clusters=n,random_state=0)
                    global y_value
                    y_value = kmeans.fit_predict(X)

            elif(combo3.get()=="Hierarchical Clustering"):

                def plot1():
                    plt.figure(figsize=(8,5),dpi=50)
                    dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
                    plt.title("Dendrogram")
                    plt.xlabel("X-Values")
                    plt.ylabel("Euclidean Distance")
                    plt.show()

                button91 = Button(window9, text="Click to view the plot", command=plot1)
                button91.place(x=290, y=80)

                label32 = Label(window9, text="Enter the number of clusters :", fg='#fa00b7', bg="#ffffff",
                                font=("Bookman Old Style", 10))
                label32.place(x=215, y=180)

                entry31 = Entry(window9)
                entry31.place(x=425, y=180, height=25, width=50)

                button92 = Button(window9, text="Next", command=lambda: [hierarchy_predict(), p.window7()])
                button92.place(x=320, y=220, width=50)

                def hierarchy_predict():
                    global y_value
                    n = int(entry31.get())
                    hc = AgglomerativeClustering(n_clusters=n,affinity="euclidean",linkage="ward")
                    y_value = hc.fit_predict(X)

        options2 = ["KMeans Clustering","Hierarchical Clustering"]
        global combo3
        combo3 = ttk.Combobox(window9, value=options2)
        combo3.config(width=50)
        combo3.current(0)
        combo3.bind("<<ComboboxSelected>>", comboclick2)
        combo3.pack()

        window9.configure(bg="#ffffff")
        window9.geometry("700x500+100+10")
        window9.title("Clustering Window")
        window9.mainloop()

    def window6(self):
        window6 = Tk()

        label61 = Label(window6, text="Features", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
        label61.place(x=150, y=100, height=30)

        z=140
        global entry61, entries
        entries = []
        for i in indep1:
            label63 = Label(window6, text=i, fg='#000000', bg="#ffffff", font=("Bookman Old Style", 15, 'bold'))
            label63.place(x=145, y=z)

            entry61 = Entry(window6)
            entry61.place(x=420, y=z, height=30, width=150)
            entries.append(entry61)
            z = z + 25

        label62 = Label(window6, text="Values", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
        label62.place(x=415, y=100)

        def predict():
            global result
            result=[]
            for entry in entries:
                result.append(entry.get())

            if(algos=="Regression"):
                global pred
                if(combo.get()=="Polynomial Regression"):
                    pred = regressor.predict(poly_reg.transform([result]))
                    label64 = Label(window6, text="Click next to view the Prediction!!!")
                    label64.place(x=260, y=z + 100)

                elif(combo.get()=="SVM Regression"):
                    pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([result])))
                    label64 = Label(window6, text="Click next to view the Prediction!!!")
                    label64.place(x=260, y=z + 100)

                else:
                    pred = regressor.predict([result])
                    label64 = Label(window6,text="Click next to view the Prediction!!!")
                    label64.place(x=260,y=z+100)

            elif(algos=="Classification"):
                if (combo2.get() == "Logistic Regression" or combo2.get()=="K-Nearest Neighbour Classification" or combo2.get()=="SVM Classification" or combo2.get()=="Decision Tree Classification" or combo2.get()=="Random Forest Classification"):
                    pred = classifier.predict(sc.transform([result]))
                    label64 = Label(window6, text="Click next to view the Prediction!!!")
                    label64.place(x=260, y=z + 100)

                elif(combo2.get()=="Naive Bayes Classification"):
                    pred = classifier.predict([result])
                    label64 = Label(window6, text="Click next to view the Prediction!!!")
                    label64.place(x=260, y=z + 100)


        button61 = Button(window6, text="Predict Model", bg="#ff0095", fg="#ffffff", command=lambda:[predict(),p.window7()])
        button61.place(x=300,y=z+50)

        window6.configure(bg="#ffffff")
        window6.geometry("700x500+100+10")
        window6.title("Prediction Window")
        window6.mainloop()

    def window7(self):
        window7 = Tk()

        if(algos=="Regression"):

            label71 = Label(window7,text="Summary", bg="#ffb700", fg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
            label71.place(x=200,y=100,width=300,height=30)

            label72 = Label(window7, text="Prediction Result", bg="#ff0095", fg="#ffffff",font=("Bookman Old Style", 15, 'bold'))
            label72.place(x=140,y=200,width=200,height=30)

            label73 = Label(window7, text="Accuracy", bg="#00ddff", fg="#ffffff", font=("Bookman Old Style", 15, 'bold'))
            label73.place(x=140, y=300, width=200, height=30)

            label74 = Label(window7, text=("{:.2f}".format(acc)), bg="#07a81f", fg="#ffffff")
            label74.place(x=460, y=300, width=100, height=30)

            pred1 = float(pred)
            pred1 = "{:.2f}".format(pred1)

            label75 = Label(window7, text=pred1, bg="#07a81f", fg="#ffffff")
            label75.place(x=460, y=200, width=100, height=30)

            button71 = Button(window7,text="Back",bg="#8c8989",fg="#ffffff",command=p.window6)
            button71.place(x=70,y=400,width=90)

            button72 = Button(window7, text="Go To Home",bg="#ff0095",fg="#ffffff",command=p.window1)
            button72.place(x=530, y=400,width=100)

        elif(algos=="Classification"):
            label71 = Label(window7, text="Summary", bg="#ffb700", fg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
            label71.place(x=200, y=100, width=300, height=30)

            label72 = Label(window7, text="Prediction Result", bg="#ff0095", fg="#ffffff",
                            font=("Bookman Old Style", 15, 'bold'))
            label72.place(x=140, y=200, width=200, height=30)

            label73 = Label(window7, text="Accuracy", bg="#00ddff", fg="#ffffff", font=("Bookman Old Style", 15, 'bold'))
            label73.place(x=140, y=300, width=200, height=30)

            label74 = Label(window7, text=("{:.2f}".format(acc)), bg="#07a81f", fg="#ffffff")
            label74.place(x=460, y=300, width=100, height=30)


            label75 = Label(window7, text=pred, bg="#07a81f", fg="#ffffff")
            label75.place(x=460, y=200, width=100, height=30)

            button71 = Button(window7, text="Back", bg="#8c8989", fg="#ffffff", command=p.window6)
            button71.place(x=70, y=400, width=90)

            button72 = Button(window7, text="Go To Home", bg="#ff0095", fg="#ffffff", command=p.window1)
            button72.place(x=530, y=400, width=100)

        elif(algos=="clustering"):
            label3 = Label(window7, text="Predicted values from clustering", fg='#18068c', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
            label3.pack(pady=20)

            frame = Frame(window7)
            frame.pack()

            scroll1 = ttk.Scrollbar(frame, orient='vertical')
            scroll1.pack(side=RIGHT, fill=Y)

            tview1 = ttk.Treeview(frame, yscrollcommand=scroll1.set)

            scroll1.config(command=tview1.yview)

            tview1["column"] = "Output"
            tview1["show"] = "headings"
            tview1.heading(tview1["column"],text=tview1["column"])

            out = y_value.tolist()

            for value in out:
                tview1.insert("",END, values=value)

            tview1.pack()

            button71 = Button(window7, text="Back", bg="#8c8989", fg="#ffffff", command=p.clustering)
            button71.place(x=70, y=400, width=90)

            button72 = Button(window7, text="Go To Home", bg="#ff0095", fg="#ffffff", command=p.window1)
            button72.place(x=530, y=400, width=100)

        window7.configure(bg="#ffffff")
        window7.geometry("700x500+100+10")
        window7.title("Output Window")
        window7.mainloop()

p = pages()
p.window1()
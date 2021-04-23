import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler
from flask import Flask, request, render_template

#Xu ly du lieu de lay ham chuan hoa
df = pd.read_csv("loans.csv")
df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
X = df.loc[:, df.columns != "not.fully.paid"].values
y = df.loc[:, df.columns == "not.fully.paid"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)
#chuan hoa du lieu
std = RobustScaler()
std.fit(X_train)

# Tải model Machine Learning
model = pickle.load(open('lr_model.pkl', 'rb'))

# Tạo ứng dụng
app = Flask(__name__)

# Liên kết hàm home với URL
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index.html')
def logo():
    return render_template('index.html')

# Liên kết chức năng dự đoán với URL
@app.route('/predict', methods=['POST'])
def predict():
    #Gom giá trị biểu mẫu vào 1 list
    features = [float(i) for i in request.form.values()]
    #Xử lý list tính năng, biến đổi giá trị phân loại bằng giá trị số sau đó chuẩn hóa dữ liệu
    a = ["credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business"]
    b = [0, 0, 0, 0, 0, 0]
    for i in range(len(a)):
        if a[i] == features[1]:
            b[i] = 1
    X = np.concatenate((features, b), axis=0)
    X = np.delete(X, 1)
    X = X.reshape(1, 18)
    X = std.transform(X)
    #Dự đoán kết quả
    prediction = model.predict(X)

    output = prediction

    #Kiểm tra các giá trị đầu ra và truy xuất kết quả bằng thẻ html dựa trên giá trị
    if output == 1:
        return render_template('index.html',
                               result='Кредит не будет погасен полностью!')
    else:
        return render_template('index.html',
                               result='Кредит будет погасен полностью!')

if __name__ == '__main__':
    #Chạy ứng dụng
    app.run()

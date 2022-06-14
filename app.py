from flask import Flask, render_template, request
from PIL import Image
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/mnist', methods = ['GET','POST'])  #도메인 /mnist로 들어갔을때는 get/post 두가지 방식으로 호출
def mnist():
    if request.method == 'GET':
        return render_template('mnist_form.html')
    else:
        f = request.files['mnistfile'] 
        # print(f.filename) # 넘긴파일이 제대로 들어왔는지 확인하기 위함.
        path = 'data/' + f.filename # 파일을 저장했다가 사용하기위해 경로 설정
        f.save(path) # 파일을 경로에 저장
        img = Image.open(path).convert('L') # 파일을 이미지로 불러와서 그레이스케일 형태로 바꿔줌
        img = np.resize(img, (1,784)) # 사이즈 조정
        img = 255-img # 반전
        # print(img.shape)
        # print(img)
        f = open('model.pickle','rb')
        model = pickle.load(f)
        f.close()
        pred = model.predict(img)
        return render_template('mnist_result.html',data = pred) # 예측값을 결과 페이지로 보냄

if __name__ == '__main__':  
    app.run(debug = True)
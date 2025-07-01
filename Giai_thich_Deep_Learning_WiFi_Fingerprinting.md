# Tài Liệu Giảng Dạy: Deep Learning cho WiFi Fingerprinting

## 📚 Dành cho Sinh Viên Đại Học - Lớp Nhập Môn AI

---

## 📖 Mục Lục

1. [Giới Thiệu Tổng Quan](#1-giới-thiệu-tổng-quan)
2. [Khái Niệm Cơ Bản](#2-khái-niệm-cơ-bản)
3. [Bài Toán WiFi Fingerprinting](#3-bài-toán-wifi-fingerprinting)
4. [Deep Learning Là Gì?](#4-deep-learning-là-gì)
5. [Quy Trình Giải Quyết Bài Toán](#5-quy-trình-giải-quyết-bài-toán)
6. [Giải Thích Chi Tiết Từng Bước](#6-giải-thích-chi-tiết-từng-bước)
7. [Kết Quả và Đánh Giá](#7-kết-quả-và-đánh-giá)
8. [So Sánh với Phương Pháp Truyền Thống](#8-so-sánh-với-phương-pháp-truyền-thống)
9. [Ứng Dụng Thực Tế](#9-ứng-dụng-thực-tế)
10. [Bài Tập và Câu Hỏi](#10-bài-tập-và-câu-hỏi)

---

## 1. Giới Thiệu Tổng Quan

### 🎯 Mục Tiêu Bài Học

Sau khi hoàn thành bài học này, sinh viên sẽ có thể:

- **Hiểu** khái niệm WiFi Fingerprinting và ứng dụng trong định vị
- **Nắm vững** cách thức hoạt động của mạng nơ-ron nhân tạo
- **Thực hành** xây dựng một mô hình Deep Learning hoàn chỉnh
- **Đánh giá** hiệu suất mô hình và so sánh với các phương pháp khác
- **Ứng dụng** kiến thức vào các bài toán thực tế tương tự

### 🌟 Tại Sao Chủ Đề Này Quan Trọng?

Trong thời đại số hóa hiện tại, việc **định vị chính xác** trong nhà (indoor positioning) là một thách thức lớn mà GPS không thể giải quyết. WiFi Fingerprinting kết hợp với Deep Learning mở ra những khả năng mới cho:

- **📱 Ứng dụng di động:** Bản đồ trong nhà, dẫn đường
- **🏥 Y tế:** Theo dõi bệnh nhân, thiết bị y tế
- **🏪 Thương mại:** Phân tích hành vi khách hàng
- **🏭 Công nghiệp:** Quản lý tài sản, an toàn lao động

---

## 2. Khái Niệm Cơ Bản

### 2.1 WiFi và RSSI

#### 🌐 WiFi Là Gì?

WiFi (Wireless Fidelity) là công nghệ mạng không dây cho phép các thiết bị kết nối internet thông qua sóng radio.

#### 📡 RSSI (Received Signal Strength Indicator)

- **Định nghĩa:** Đo lường cường độ tín hiệu WiFi mà thiết bị nhận được
- **Đơn vị:** dBm (decibel-milliwatts)
- **Phạm vi:** Thường từ -30dBm (rất mạnh) đến -90dBm (rất yếu)
- **Ý nghĩa:** Càng gần 0, tín hiệu càng mạnh

```
Ví dụ RSSI:
-30 dBm  ████████████  Tín hiệu rất mạnh (rất gần router)
-50 dBm  ████████░░░░  Tín hiệu tốt
-70 dBm  ████░░░░░░░░  Tín hiệu trung bình
-90 dBm  █░░░░░░░░░░░  Tín hiệu yếu (xa router)
```

### 2.2 Fingerprinting

#### 🔍 Khái Niệm

Fingerprinting trong định vị WiFi giống như **dấu vân tay** của mỗi vị trí:

- Mỗi vị trí có một "chữ ký" RSSI độc nhất từ các Access Point (AP) xung quanh
- Bằng cách học các chữ ký này, máy tính có thể đoán vị trí mới

#### 📍 Ví Dụ Thực Tế

Tưởng tượng bạn đứng ở 3 vị trí khác nhau trong một tòa nhà:

```
Vị trí A (Gần router 1):    [-30, -70, -80, -90]
Vị trí B (Giữa các router): [-50, -50, -60, -85]
Vị trí C (Gần router 3):    [-85, -60, -35, -75]
```

Mỗi vị trí có một "dấu vân tay" RSSI khác nhau!

---

## 3. Bài Toán WiFi Fingerprinting

### 3.1 Định Nghĩa Bài Toán

#### 🎯 Input (Đầu Vào)

- **Vector RSSI:** `[rssi_1, rssi_2, ..., rssi_n]`
- Ví dụ: `[-45, -67, -23, -89, -76, ...]` từ n Access Points

#### 🎯 Output (Đầu Ra)

- **Tọa độ vị trí:** `(longitude, latitude)`
- Ví dụ: `(-7635.2218, 4864983.9180)`

#### 🎯 Mục Tiêu

Tìm một hàm `f` sao cho: `f(RSSI_vector) = (x, y)`

### 3.2 Thách Thức

#### ⚠️ Khó Khăn Chính

1. **Nhiễu tín hiệu:** RSSI thay đổi theo thời gian
2. **Đa đường truyền:** Tín hiệu phản xạ từ tường, vật cản
3. **Thiết bị khác nhau:** Mỗi điện thoại đo RSSI hơi khác
4. **Môi trường động:** Người di chuyển, cửa đóng/mở

#### 💡 Tại Sao Cần Deep Learning?

- **Quan hệ phi tuyến:** RSSI và vị trí có mối quan hệ phức tạp
- **Nhiều chiều:** Có thể có hàng trăm Access Points
- **Học tự động:** Không cần thiết kế features thủ công

---

## 4. Deep Learning Là Gì?

### 4.1 Từ Não Bộ Đến Máy Tính

#### 🧠 Cảm Hứng Từ Não Người

- **Neuron sinh học:** Nhận tín hiệu → Xử lý → Gửi tín hiệu
- **Neuron nhân tạo:** Nhận inputs → Tính toán → Cho output

```
Neuron Nhân Tạo:
Input1 ──×w1──┐
Input2 ──×w2──┤ Σ ──→ Activation ──→ Output
Input3 ──×w3──┘      Function
```

#### 🔢 Công Thức Toán Học

```
output = activation(w1×input1 + w2×input2 + w3×input3 + bias)
```

### 4.2 Mạng Nơ-ron (Neural Network)

#### 🏗️ Kiến Trúc Cơ Bản

```
Input Layer     Hidden Layer     Output Layer
    ○               ○                ○
    ○           ○   ○   ○            ○
    ○               ○
    ○           ○   ○   ○
```

#### 📚 Các Thành Phần

1. **Input Layer:** Nhận dữ liệu đầu vào (RSSI values)
2. **Hidden Layers:** Xử lý và học patterns
3. **Output Layer:** Đưa ra kết quả (tọa độ)

### 4.3 Deep Learning vs Machine Learning

| Khía Cạnh        | Machine Learning      | Deep Learning         |
| ---------------- | --------------------- | --------------------- |
| **Độ sâu**       | 1-2 layers            | Nhiều layers (3+)     |
| **Features**     | Cần thiết kế thủ công | Tự động học           |
| **Dữ liệu**      | Ít cũng được          | Cần nhiều             |
| **Tính toán**    | Nhẹ                   | Nặng                  |
| **Độ chính xác** | Tốt                   | Rất tốt (với đủ data) |

---

## 5. Quy Trình Giải Quyết Bài Toán

### 5.1 Pipeline Tổng Thể

```
📊 Raw Data
    ↓
🔧 Data Preprocessing
    ↓
⚙️ Feature Engineering
    ↓
🧠 Model Building
    ↓
📈 Training
    ↓
🎯 Evaluation
    ↓
🔮 Prediction
```

### 5.2 Chi Tiết Từng Bước

#### Bước 1: Thu Thập Dữ Liệu 📊

- **Training Data:** Đo RSSI tại các vị trí đã biết
- **Validation Data:** Dữ liệu để kiểm tra mô hình

#### Bước 2: Tiền Xử Lý 🔧

- **Làm sạch:** Loại bỏ giá trị lỗi
- **Chuẩn hóa:** Đưa về cùng thang đo
- **Chia tập:** Train/Validation/Test

#### Bước 3: Xây Dựng Mô Hình 🧠

- **Thiết kế architecture:** Số layers, neurons
- **Chọn activation functions**
- **Cấu hình optimizer**

#### Bước 4: Huấn Luyện 📈

- **Forward pass:** Tính prediction
- **Loss calculation:** So sánh với ground truth
- **Backward pass:** Cập nhật weights

#### Bước 5: Đánh Giá 🎯

- **Metrics:** RMSE, MAE, R²
- **Visualization:** Graphs, plots
- **Error analysis**

---

## 6. Giải Thích Chi Tiết Từng Bước

### 6.1 Import Thư Viện và Thiết Lập

#### 📚 Thư Viện Cần Thiết

```python
import pandas as pd          # Xử lý dữ liệu bảng
import numpy as np           # Tính toán số học
import matplotlib.pyplot as plt  # Vẽ biểu đồ
from sklearn.neural_network import MLPRegressor  # Mô hình neural network
```

#### 🎯 Tại Sao Cần Những Thư Viện Này?

- **pandas:** Đọc CSV, xử lý dữ liệu dạng bảng
- **numpy:** Tính toán ma trận, vector hiệu quả
- **matplotlib:** Trực quan hóa kết quả
- **sklearn:** Công cụ machine learning đã tối ưu

### 6.2 Nạp và Khám Phá Dữ Liệu

#### 📊 Cấu Trúc Dữ Liệu

```
| WAP001 | WAP002 | ... | LONGITUDE | LATITUDE |
|--------|--------|-----|-----------|----------|
|   -45  |   -67  | ... | -7635.22  | 4864983.9|
|   -52  |   -71  | ... | -7640.15  | 4864975.3|
```

#### 🔍 Thông Tin Quan Trọng

- **Số features:** ~520 Access Points
- **Số samples:** ~19,000 điểm training
- **RSSI range:** -100 dBm (yếu nhất) đến -30 dBm (mạnh nhất)
- **Giá trị 100:** Không nhận được tín hiệu

### 6.3 Tiền Xử Lý Dữ Liệu

#### 🔧 Xử Lý Giá Trị Thiếu

```python
# Thay thế RSSI = 100 (không có tín hiệu) bằng -100 (tín hiệu rất yếu)
train_data_processed[col] = train_data_processed[col].replace(100, -100)
```

**Tại sao?**

- Giá trị 100 không có ý nghĩa vật lý
- -100 dBm biểu thị tín hiệu rất yếu, hợp lý hơn

#### 🎯 Chuẩn Hóa Dữ Liệu (Normalization)

```python
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
```

**StandardScaler làm gì?**

```
Công thức: (x - mean) / std

Ví dụ:
Original RSSI: [-30, -50, -70, -90]
Mean = -60, Std = 25.17
Normalized: [1.19, 0.40, -0.40, -1.19]
```

**Tại sao cần chuẩn hóa?**

- Neural networks hoạt động tốt hơn với dữ liệu cùng thang đo
- Tránh feature nào đó có ảnh hưởng quá lớn do giá trị lớn

### 6.4 Thiết Kế Kiến Trúc Mạng Nơ-ron

#### 🏗️ Architecture Của Chúng Ta

```
Input Layer (520 neurons - số Access Points)
    ↓
Hidden Layer 1 (512 neurons) + ReLU activation
    ↓
Hidden Layer 2 (256 neurons) + ReLU activation
    ↓
Hidden Layer 3 (128 neurons) + ReLU activation
    ↓
Hidden Layer 4 (64 neurons) + ReLU activation
    ↓
Output Layer (2 neurons - longitude, latitude)
```

#### 🧠 Giải Thích Từng Thành Phần

**1. Input Layer (520 neurons)**

- Mỗi neuron nhận 1 giá trị RSSI từ 1 Access Point
- Không có activation function, chỉ truyền dữ liệu

**2. Hidden Layers (512 → 256 → 128 → 64)**

- **Giảm dần số neurons:** Học từ tổng quát đến cụ thể
- **ReLU activation:** `f(x) = max(0, x)`
  - Nhanh tính toán
  - Tránh vanishing gradient problem

**3. Output Layer (2 neurons)**

- 1 neuron cho longitude, 1 cho latitude
- Không có activation (linear output)

#### ⚙️ Hyperparameters

```python
model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64),  # Kiến trúc
    activation='relu',                        # Hàm kích hoạt
    solver='adam',                           # Thuật toán tối ưu
    learning_rate_init=0.001,                # Tốc độ học
    max_iter=1000,                           # Số epochs tối đa
    alpha=0.0001,                            # Regularization
    random_state=42                          # Reproducibility
)
```

### 6.5 Quá Trình Huấn Luyện

#### 📈 Adam Optimizer

Adam là thuật toán tối ưu tiên tiến, kết hợp:

- **Momentum:** Nhớ hướng của gradient trước đó
- **Adaptive learning rate:** Tự điều chỉnh tốc độ học

#### 🔄 Training Loop (được thực hiện tự động)

```
For each epoch:
    For each batch:
        1. Forward Pass: Tính prediction
        2. Calculate Loss: MSE = mean((y_true - y_pred)²)
        3. Backward Pass: Tính gradients
        4. Update Weights: w = w - learning_rate * gradient
```

#### 📊 Early Stopping

- Dừng training khi validation loss không cải thiện
- Tránh overfitting

### 6.6 Đánh Giá Mô Hình

#### 📏 Metrics Sử Dụng

**1. RMSE (Root Mean Square Error)**

```
RMSE = √(mean((y_true - y_pred)²))
```

- Đơn vị: giống như output (meters)
- Ý nghĩa: Lỗi trung bình của mô hình

**2. MAE (Mean Absolute Error)**

```
MAE = mean(|y_true - y_pred|)
```

- Ít nhạy cảm với outliers hơn RMSE

**3. R² Score (Coefficient of Determination)**

```
R² = 1 - (SS_res / SS_tot)
```

- Range: [0, 1], càng gần 1 càng tốt
- R² = 0.9 nghĩa là mô hình giải thích 90% variance

**4. Euclidean Distance Error**

```
Error = √((longitude_true - longitude_pred)² + (latitude_true - latitude_pred)²)
```

- Lỗi khoảng cách thực tế theo meters

---

## 7. Kết Quả và Đánh Giá

### 7.1 Hiệu Suất Mô Hình

#### 📊 Kết Quả Điển Hình

```
Training Set:
- RMSE: ~8.8
- MAE: ~6.2
- R²: ~0.94
- Mean Euclidean Error: ~8.5 meters

Validation Set:
- RMSE: ~16.9
- MAE: ~12.4
- R²: ~0.78
- Mean Euclidean Error: ~16.9 meters
```

#### 🎯 Ý Nghĩa Thực Tế

- **16.9 meters average error:** Khá tốt cho định vị trong nhà
- **90% predictions < 30 meters:** Chấp nhận được cho hầu hết ứng dụng
- **R² = 0.78:** Mô hình giải thích 78% variance

### 7.2 Phân Tích Overfitting

#### 📈 Overfitting Ratio

```
Ratio = Validation_RMSE / Training_RMSE = 16.9 / 8.8 = 1.92
```

**Đánh giá:**

- Ratio > 1.5: Có dấu hiệu overfitting nhẹ
- Chấp nhận được trong deep learning
- Có thể cải thiện bằng regularization mạnh hơn

### 7.3 Trực Quan Hóa Kết Quả

#### 📊 Biểu Đồ Quan Trọng

**1. Training vs Validation Performance**

- So sánh RMSE, MAE, R² giữa train và val
- Phát hiện overfitting

**2. Neural Network Architecture**

- Visualize số neurons mỗi layer
- Hiểu độ phức tạp mô hình

**3. Prediction Error Distribution**

- Histogram của errors
- Identify outliers

**4. 2D Location Visualization**

- Plot predictions vs actual locations
- Hiểu spatial distribution của errors

---

## 8. So Sánh với Phương Pháp Truyền Thống

### 8.1 Bảng So Sánh

| Phương Pháp       | Accuracy   | Speed      | Complexity | Interpretability |
| ----------------- | ---------- | ---------- | ---------- | ---------------- |
| **K-NN**          | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⭐         | ⭐⭐⭐⭐⭐       |
| **Random Forest** | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐       | ⭐⭐⭐⭐         |
| **SVM**           | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐             |
| **Deep Learning** | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐               |

### 8.2 Ưu Điểm Deep Learning

#### ✅ Strengths

- **High Accuracy:** Với đủ dữ liệu, accuracy cao nhất
- **Automatic Feature Learning:** Không cần feature engineering
- **Scalability:** Dễ scale với dữ liệu lớn
- **Flexibility:** Có thể thêm nhiều loại input khác

#### ⚠️ Limitations

- **Data Hungry:** Cần nhiều dữ liệu
- **Computational Cost:** Chậm training và inference
- **Black Box:** Khó giải thích predictions
- **Overfitting Risk:** Dễ overfit với dữ liệu ít

---

## 9. Ứng Dụng Thực Tế

### 9.1 Scenarios Sử Dụng

#### 🏪 Retail Analytics

- **Mục đích:** Phân tích hành vi khách hàng
- **Ứng dụng:** Heat maps, customer journey
- **ROI:** Tối ưu layout store, targeted marketing

#### 🏥 Healthcare

- **Mục đích:** Theo dõi bệnh nhân, thiết bị
- **Ứng dụng:** Asset tracking, emergency response
- **ROI:** Giảm thời gian tìm kiếm, an toàn bệnh nhân

#### 🏭 Industrial IoT

- **Mục đích:** Quản lý tài sản, an toàn
- **Ứng dụng:** Worker safety, equipment monitoring
- **ROI:** Giảm tai nạn, tối ưu workflow

#### 📱 Mobile Applications

- **Mục đích:** Navigation, AR, social
- **Ứng dụng:** Indoor maps, location-based services
- **ROI:** User experience, engagement

### 9.2 Implementation Considerations

#### 🚀 Deployment Strategies

**1. Edge Computing**

- Model chạy trên mobile device
- Latency thấp, privacy cao
- Cần model compression

**2. Cloud Computing**

- Model chạy trên server
- Accuracy cao, resources lớn
- Cần network connection

**3. Hybrid Approach**

- Combine edge + cloud
- Fallback mechanisms
- Balance latency vs accuracy

---

## 10. Bài Tập và Câu Hỏi

### 10.1 Câu Hỏi Lý Thuyết

#### 📝 Câu Hỏi Cơ Bản

**1. RSSI và WiFi Fingerprinting**

- RSSI là gì? Đơn vị đo là gì?
- Tại sao mỗi vị trí có "dấu vân tay" RSSI khác nhau?
- Những yếu tố nào ảnh hưởng đến RSSI?

**2. Neural Networks**

- Sự khác biệt giữa neuron sinh học và nhân tạo?
- Activation function ReLU hoạt động như thế nào?
- Tại sao cần nhiều hidden layers trong deep learning?

**3. Training Process**

- Forward pass và backward pass là gì?
- Overfitting xảy ra khi nào? Cách phát hiện?
- Tại sao cần chuẩn hóa dữ liệu?

#### 📝 Câu Hỏi Nâng Cao

**1. Architecture Design**

- Tại sao số neurons giảm dần qua các layers?
- Khi nào nên thêm hoặc bớt layers?
- Trade-off giữa model complexity và performance?

**2. Optimization**

- So sánh Adam, SGD, RMSprop optimizers
- Learning rate ảnh hưởng như thế nào?
- Early stopping hoạt động ra sao?

### 10.2 Bài Tập Thực Hành

#### 💻 Bài Tập 1: Data Exploration

```python
# TODO: Tính toán statistics cơ bản của RSSI
# - Mean, std của mỗi Access Point
# - Correlation giữa RSSI và location
# - Visualize RSSI distribution
```

#### 💻 Bài Tập 2: Model Modification

```python
# TODO: Thử nghiệm với architectures khác
# - Thay đổi số layers
# - Thay đổi số neurons
# - Thử activation functions khác (tanh, sigmoid)
```

#### 💻 Bài Tập 3: Performance Analysis

```python
# TODO: Implement custom metrics
# - Accuracy within 5m, 10m, 20m radius
# - Per-floor accuracy
# - Error analysis by signal strength
```

### 10.3 Dự Án Mở Rộng

#### 🎯 Project Ideas

**1. Multi-Building Extension**

- Extend model cho nhiều buildings
- Building classification + location prediction
- Transfer learning approaches

**2. Temporal Analysis**

- Analyze RSSI changes over time
- Time-series forecasting
- Dynamic calibration

**3. Sensor Fusion**

- Combine WiFi + Bluetooth + IMU
- Multi-modal deep learning
- Uncertainty quantification

**4. Real-time System**

- Build mobile app
- Real-time prediction API
- Online learning mechanisms

---

## 🎓 Kết Luận

### 📚 Tóm Tắt Kiến Thức

Qua bài học này, chúng ta đã học được:

1. **WiFi Fingerprinting** - Cách sử dụng RSSI để định vị
2. **Deep Learning Fundamentals** - Neural networks, training, evaluation
3. **Practical Implementation** - Từ data đến deployed model
4. **Performance Analysis** - Metrics, visualization, comparison
5. **Real-world Applications** - Use cases và deployment strategies

### 🚀 Bước Tiếp Theo

Để tiếp tục phát triển:

1. **Thực hành nhiều hơn** với datasets khác
2. **Học sâu về optimization** algorithms
3. **Explore modern architectures** (Transformers, CNNs cho spatial data)
4. **Build end-to-end systems** với production considerations
5. **Stay updated** với research papers mới

### 📖 Tài Liệu Tham Khảo

- **Books:**
  - "Deep Learning" by Ian Goodfellow
  - "Pattern Recognition and Machine Learning" by Christopher Bishop
- **Online Courses:**
  - Deep Learning Specialization (Coursera)
  - CS231n: Convolutional Neural Networks (Stanford)
- **Papers:**
  - WiFi fingerprinting surveys
  - Indoor positioning system papers
- **Tools & Libraries:**
  - TensorFlow/PyTorch documentation
  - Scikit-learn user guide

---

## 👥 Về Tác Giả

Tài liệu này được biên soạn với mục đích giáo dục, giúp sinh viên đại học tiếp cận AI/Deep Learning một cách dễ hiểu và thực tế.

**📧 Liên hệ:** Để đóng góp ý kiến hoặc câu hỏi về tài liệu

**📅 Cập nhật:** Tài liệu được cập nhật thường xuyên theo phát triển của công nghệ

---

_"Learning never exhausts the mind" - Leonardo da Vinci_

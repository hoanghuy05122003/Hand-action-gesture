# ✋ Hand Action Gesture Recognition (3D-CNN + RNN/LSTM/GRU)

> Nhận dạng cử chỉ tay động từ video bằng mạng **3D-CNN kết hợp RNN / LSTM / GRU**, huấn luyện trên 6 lớp của bộ dữ liệu **20BN-Jester**, kèm ứng dụng desktop **PyQt6** nhận dạng thời gian thực qua webcam.

Đồ án tốt nghiệp – Hoàng Quang Huy, lớp 12421TN, Trường Đại học Sư phạm Kỹ thuật Hưng Yên. Báo cáo: [`BCDATN_Hoàng Quang Huy_12421TN_10121840.pdf`](<BCDATN_Hoàng Quang Huy_12421TN_10121840.pdf>) và bản Word [`Đồ án tốt nghiệp Hoàng Quang Huy.docx`](<Đồ án tốt nghiệp Hoàng Quang Huy.docx>).

| Hạng mục | Nội dung |
|:---|:---|
| Dữ liệu | 20BN-Jester, 6 lớp: *No gesture, Shaking Hand, Swiping Up, Thumb Down, Thumb Up, Zooming Out With Two Fingers* |
| Số mẫu | 9.912 train · 1.239 validation · 1.239 test (video đã tách khung) |
| Đầu vào | Chuỗi **37 khung** ảnh xám **100×100** → tensor `(37, 100, 100, 1)` |
| Kiến trúc | 4 khối Conv3D (32→64→128→256) + BatchNorm + MaxPool3D → TimeDistributed(Flatten, Dense 128, Dropout) → RNN / LSTM / GRU 128 → Softmax 6 lớp |
| Mô hình tốt nhất | **3D-CNN + LSTM**, test accuracy **95,39%** (macro F1 0,95) |
| Demo | PyQt6 + OpenCV, dự đoán theo từng cửa sổ 37 khung từ webcam |

---

## 1. Kiến trúc mô hình

```
Video (37 khung, xám, 100×100)
   ▼
Conv3D(32, 3×3×3) → BatchNorm → MaxPool3D(2×2×2)
Conv3D(64, 3×3×3) → BatchNorm → MaxPool3D(2×2×2)      ← trích đặc trưng không gian–thời gian
Conv3D(128,3×3×3) → BatchNorm → MaxPool3D(2×2×2)        (L2 = 0,001 trên mỗi Conv3D)
Conv3D(256,3×3×3) → BatchNorm → MaxPool3D(2×2×2)
   ▼
TimeDistributed(Flatten) → TimeDistributed(Dense 128, ReLU) → TimeDistributed(Dropout 0,2)
   ▼
SimpleRNN / LSTM / GRU (128)                            ← mô hình hóa chuỗi theo thời gian
   ▼
Dense(128, ReLU) → Dropout(0,5) → Dense(6, Softmax)
```

Khoảng 2,4–2,5 triệu tham số tùy biến thể. Dữ liệu được nạp bằng `VideoDataGenerator` (kế thừa `keras.utils.Sequence`) đọc từng thư mục khung theo file CSV, chuẩn hóa về [0, 1] và one-hot nhãn.

Huấn luyện: Adam, categorical cross-entropy, batch 32, `EarlyStopping(val_loss, patience=5, restore_best_weights)` và `ReduceLROnPlateau(val_loss, patience=3)`.

## 2. Kết quả thực nghiệm

Ba cấu hình huấn luyện được thử, mỗi cấu hình chạy với cả 3 loại lớp hồi quy. Accuracy trên tập test:

| Cấu hình | 3D-CNN + RNN | 3D-CNN + LSTM | 3D-CNN + GRU |
|:---|:---:|:---:|:---:|
| `factor0.1-20epochs` (ReduceLR factor 0,1, 20 epoch) | 86,68% | 92,02% | 92,11% |
| `factor0.1-25epochs` (factor 0,1, 25 epoch) | 88,32% | 92,35% | 93,01% |
| `factor0.5-25epochs-midelta0.001` (factor 0,5, min_delta 0,001, 25 epoch) | 92,52% | **95,39%** | 94,00% |

Chi tiết từng lớp của mô hình tốt nhất (3D-CNN + LSTM, cấu hình factor 0,5):

| Lớp | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|
| No gesture | 0,98 | 0,99 | 0,99 |
| Shaking Hand | 1,00 | 1,00 | 1,00 |
| Swiping Up | 0,93 | 0,93 | 0,93 |
| Thumb Down | 0,95 | 0,96 | 0,96 |
| Thumb Up | 0,90 | 0,93 | 0,91 |
| Zooming Out With Two Fingers | 0,97 | 0,92 | 0,95 |
| **Accuracy** | | | **0,95** |

Nhận xét: LSTM và GRU vượt rõ SimpleRNN nhờ giữ được phụ thuộc dài hạn giữa các khung; giảm learning rate nhẹ hơn (factor 0,5) giúp cả ba biến thể hội tụ tốt hơn.

## 3. Ứng dụng demo (PyQt6)

`demo/app.py` mở webcam (`cv2.VideoCapture(0)`), mỗi khung được chuyển sang ảnh xám 100×100 và đưa vào bộ đệm. Khi đủ 37 khung, chuỗi được chuẩn hóa và đưa vào model `.keras`, giao diện hiển thị tên cử chỉ (kèm emoji) cùng độ tin cậy, sau đó bộ đệm được làm mới cho cửa sổ tiếp theo. Có nút **Start / Stop** camera và giao diện tối.

```bash
pip install tensorflow opencv-python PyQt6 numpy
cd demo
python app.py
```

## 4. Cấu trúc thư mục

```
Hand-action-gesture/
├── factor0.1-20epochs/                 # Thí nghiệm 1: 3dcnn_rnn, 3dcnn+lstm, 3dcnn+gru
├── factor0.1-25epochs/                 # Thí nghiệm 2
├── factor0.5-25epochs-midelta0.001/    # Thí nghiệm 3 (cho kết quả tốt nhất)
├── demo/
│   ├── app.py                          # Ứng dụng nhận dạng realtime PyQt6
│   └── *.keras, *.h5                   # Model 3D-CNN + LSTM đã huấn luyện
├── BCDATN_Hoàng Quang Huy_12421TN_10121840.pdf
└── Đồ án tốt nghiệp Hoàng Quang Huy.docx
```

Các notebook được chạy trên Google Colab (dữ liệu giải nén từ Google Drive); cần sửa đường dẫn `data_dir` khi chạy ở môi trường khác. Bộ dữ liệu không đi kèm repo.

## 5. Công nghệ

Python · TensorFlow / Keras (Conv3D, LSTM, GRU, SimpleRNN) · OpenCV · NumPy / Pandas · scikit-learn · Matplotlib · PyQt6 · Google Colab

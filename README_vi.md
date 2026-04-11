# Khung Hệ thống Phát hiện Ung thư Phổi

**Hệ thống hỗ trợ chẩn đoán dựa trên nền tảng web, tích hợp Học sâu và thiết kế hướng tới lâm sàng**

---

## Tóm tắt (Abstract)

Kho mã nguồn này trình bày một **hệ thống hỗ trợ chẩn đoán trên nền tảng web**, được thiết kế nhằm hỗ trợ phát hiện sớm các bất thường phổi, đặc biệt là **ung thư phổi**, thông qua phân tích hình ảnh X-quang lồng ngực. Kiến trúc hệ thống kết hợp backend hiệu năng cao dựa trên **FastAPI** với frontend gọn nhẹ, có tính tương tác cao.

Thông qua việc sử dụng **GSAP (GreenSock Animation Platform)**, hệ thống đảm bảo trải nghiệm tương tác người–máy (Human–Computer Interaction, HCI) mượt mà và trực quan. Lõi suy luận (inference engine) được xây dựng dựa trên các phương pháp **Học sâu (Deep Learning)**, triển khai bằng **TensorFlow** và **PyTorch**, nhằm phân tích dữ liệu hình ảnh y sinh phục vụ bài toán chẩn đoán.

> **Ghi chú:** Phiên bản tài liệu gốc bằng tiếng Anh có thể được tham khảo [tại đây](README.md).

---

## 1. Năng lực Hệ thống và Kiến trúc

Ứng dụng được thiết kế theo kiến trúc nhiều tầng (multi-tier architecture), phù hợp cho cả mục đích nghiên cứu và triển khai thử nghiệm, đảm bảo khả năng mở rộng, bảo trì và tái sử dụng.

### Các chức năng cốt lõi

* **Xử lý ảnh X-quang y khoa**
  Tự động tiếp nhận, tiền xử lý và chuẩn hóa dữ liệu hình ảnh X-quang đầu vào để phục vụ suy luận chẩn đoán.

* **Giao diện người dùng có độ trung thực cao**
  Giao diện tương tác được xây dựng với **GSAP**, cho phép thiết kế chuyển động tinh vi, cải thiện khả năng tiếp nhận thông tin và phản hồi thị giác.

* **Backend bất đồng bộ, hiệu năng cao**
  Sử dụng **FastAPI** nhằm tối ưu độ trễ, hỗ trợ xử lý đồng thời và đáp ứng tốt các yêu cầu thời gian thực.

* **Tích hợp mô hình Học sâu**
  Triển khai các mô hình mạng nơ-ron tích chập (Convolutional Neural Networks – CNNs) thông qua **TensorFlow** và **PyTorch** cho các bài toán phân loại nhị phân hoặc đa lớp.

### Dữ liệu sử dụng

Quá trình xây dựng, huấn luyện và đánh giá mô hình tham chiếu bộ dữ liệu X-quang mã nguồn mở:

* **Nguồn dữ liệu:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


### Thẩm định Lâm sàng & Tuân thủ Pháp quy (Tiêu chuẩn FDA/CE)
Nhằm đáp ứng các tiêu chuẩn quốc tế về thiết bị y tế (như FDA 21 CFR hoặc dấu CE đối với Phần mềm dưới dạng Thiết bị Y tế - SaMD), khung vận hành tích hợp các chỉ số chẩn đoán sau:

- Quyết định Nhị phân (Binary Decision): Phân loại danh mục dứt khoát dựa trên suy luận của mô hình.

- Điểm số Quyết định (Decision Score): Kết quả dạng số thô thể hiện giá trị đánh giá nội bộ của mô hình.

- Ngưỡng phê duyệt (Approved Threshold): Điểm cắt (cut-off) đã hiệu chuẩn, được tối ưu hóa cho độ nhạy và độ đặc hiệu lâm sàng.

- Xác suất Rủi ro (Prisk​): Khả năng thống kê được định lượng về sự hiện diện của bệnh lý.

- Độ tin cậy Dự đoán (Prediction Confidence): Chỉ số cho biết mức độ chắc chắn đối với kết luận của thuật toán.

- Miễn trừ trách nhiệm pháp quy: Hệ thống này được phân loại là Chỉ hỗ trợ bởi AI; mục đích nhằm hỗ trợ bác sĩ lâm sàng và không được sử dụng như một công cụ chẩn đoán độc lập mà không có sự giám sát chuyên môn y tế.
---

## 2. Ngăn xếp Công nghệ (Technology Stack)

| Thành phần          | Công nghệ                 | Mô tả                         |
| :------------------ | :------------------------ | :---------------------------- |
| **Môi trường chạy** | Python 3.9+               | Nền tảng chính cho backend    |
| **Framework API**   | FastAPI                   | Framework web hiệu năng cao   |
| **Máy chủ**         | Uvicorn                   | ASGI server cho FastAPI       |
| **Học máy**         | TensorFlow / PyTorch      | Thư viện Học sâu cho suy luận |
| **Frontend**        | HTML5 / CSS3 / JavaScript | Xây dựng giao diện người dùng |
| **Hiệu ứng**        | GSAP                      | Thư viện chuyển động nâng cao |

---

## 3. Điều kiện Tiên quyết

Trước khi triển khai hệ thống, môi trường cần đáp ứng các yêu cầu sau:

1. **Python** phiên bản **3.9** trở lên

   * Linux / macOS: `python3 --version`
   * Windows: `python --version`

2. **Git** (khuyến nghị) để quản lý mã nguồn và phiên bản.

---

## 4. Quy trình Cài đặt

### Bước 1: Tải mã nguồn

Sao chép kho mã nguồn về máy cục bộ bằng Git hoặc tải trực tiếp từ GitHub:

```bash
git clone https://github.com/baokhanh546123/API_LungCancer
```

### Bước 2: Cấu hình môi trường ảo

Việc sử dụng môi trường ảo (**venv**) được khuyến nghị mạnh mẽ nhằm cô lập các phụ thuộc của dự án.

#### Windows

```bash
cd API_LungCancer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS

```bash
cd API_LungCancer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Quy trình Vận hành

### Chế độ Phát triển (Development Mode)

Phù hợp cho quá trình nghiên cứu, thử nghiệm và gỡ lỗi. Máy chủ sẽ tự động tải lại khi mã nguồn thay đổi.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Chế độ Triển khai (Production Mode)

Sử dụng cho các kịch bản chạy ổn định, không yêu cầu tải lại nóng.

* **Windows**

```bash
python main.py
```

* **Linux / macOS**

```bash
python3 main.py
```

Sau khi khởi động thành công, hệ thống có thể được truy cập tại:

```
http://127.0.0.1:8000/
```

---

## 6. Đóng góp và Phản hồi

Dự án được mở cho các đóng góp mang tính học thuật và kỹ thuật. Mọi phản hồi liên quan đến độ chính xác chẩn đoán, khả năng mở rộng kiến trúc hoặc trải nghiệm người dùng đều được đánh giá cao.

* **Quản lý vấn đề & đóng góp mã nguồn:** Thông qua [GitHub Repository](https://github.com/baokhanh546123/API_LungCancer)
* **Liên hệ trực tiếp:** [Tại Đây](mailto:tranphucdangkhanh1011dl@gmail.com)

Những đóng góp của cộng đồng đóng vai trò quan trọng trong việc hoàn thiện và nâng cao giá trị nghiên cứu của hệ thống.

---

## 7. Thông tin Tác giả

* **Tác giả:** TS Võ Phương Bình , Sinh viên Nguyễn Thị Ngọc Nhi , Trần Phúc Đăng Khánh.
* **Đơn vị:** Trường Đại học Đà Lạt
* **Thời điểm phát hành:** 27/12/2025

---

## 8. Tuyên bố và Bản quyền

Phần mềm này được phát triển **phục vụ mục đích học tập và nghiên cứu khoa học**.

1. **Ghi nguồn:** Mọi hình thức sao chép, chỉnh sửa hoặc phân phối lại, toàn bộ hoặc một phần, đều phải được sự cho phép của tác giả và ghi rõ nguồn gốc.
2. **Phi thương mại:** Nghiêm cấm sử dụng dự án cho các mục đích thương mại khi chưa có sự đồng ý bằng văn bản từ tác giả.

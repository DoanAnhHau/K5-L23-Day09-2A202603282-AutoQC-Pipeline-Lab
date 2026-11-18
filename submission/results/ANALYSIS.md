# Phân tích kết quả train: raw và clean

## Phạm vi

Bốn kết quả dùng YOLO26n, 20 epoch, batch 16, imgsz 640, cùng cấu hình và test 100 ảnh; mỗi biến thể có seed 0 và 1. **clean** là tên biến thể của bản xuất clean_v1; pipeline đang kết luận **Hold**, nên tên này không có nghĩa dữ liệu đã được xác nhận sạch hoàn toàn.

## 1. Chênh mAP50 và độ lệch giữa seed

| Biến thể | Seed 0 | Seed 1 | Trung bình ± SD (n=2) |
|---|---:|---:|---:|
| raw | 0.7256 | 0.7259 | 0.7258 ± 0.0002 (0.726 ± 0.000 khi làm tròn) |
| clean | 0.6813 | 0.7163 | 0.6988 ± 0.0247 (0.699 ± 0.025 khi làm tròn) |

Chênh trung bình clean − raw là **−0.0270**, tức clean thấp hơn 2.70 điểm phần trăm mAP50 trong các lượt này. Độ lớn chênh lệch bằng khoảng 1.09 lần SD của clean. Theo từng seed, clean thấp hơn raw 0.0443 ở seed 0 và 0.0096 ở seed 1.

Với hai seed, đây chỉ là mô tả bốn lượt train, không đủ để kết luận tổng quát raw tốt hơn clean. SD raw gần 0 chỉ nói hai seed đã chạy tình cờ gần nhau; nó không chứng minh raw ổn định nói chung. Kết luận thận trọng: **clean chưa cho thấy mAP50 cao hơn raw trong lần so sánh này**. Chênh lệch không thể quy riêng cho thao tác làm sạch vì số ảnh, nhãn và thành phần dữ liệu cũng thay đổi. Clean có precision trung bình cao hơn khoảng 0.032, nhưng recall thấp hơn 0.053 và mAP50-95 thấp hơn 0.026; nên xem cả các chỉ số thay vì chỉ mAP50.

## 2. Raw thực sự train trên bao nhiêu ảnh?

Hai JSON raw ghi train_images: 224. Trong scripts/train_compare.py, con số này là số file trong work/train/raw/train/images được đếm sau train, tức số ảnh được chuẩn bị/cấp cho Ultralytics trước khi nó lọc mẫu.

Notebook Colab đã lưu output quét nhãn raw: 224/224 ảnh được quét; Ultralytics báo 11 mẫu corrupt và in 11 dòng “ignoring corrupt image/label”. Vì vậy **213 ảnh còn hợp lệ để train** (224 − 11). Các ảnh bị bỏ là `img_0010`, `img_0014`, `img_0020`, `img_0022`, `img_0099`, `img_0104`, `img_0117`, `img_0136`, `img_0179`, `img_0194`, `img_0199`. 19 ảnh nền được báo trong dòng scan không phải 19 ảnh bị loại: ảnh nền vẫn là mẫu train hợp lệ, chỉ không có box. Các mẫu hỏng gồm ảnh không đọc được, tọa độ không chuẩn/ngoài biên, giá trị nhãn âm, class ID ngoài 0–3 và một dòng nhãn sai cấu trúc. Do đó cần phân biệt số JSON 224 (ảnh cấp vào) với số sau lọc 213.

Đối chiếu thêm, notebook ghi clean quét 203 ảnh, 0 corrupt và 4 ảnh nền; cả 203 ảnh hợp lệ đều có thể tham gia train. Các số này khớp với số file trong JSON, vì clean không bị Ultralytics loại thêm.

Pipeline v1 ghi 3 BAD_FORMAT, 6 BAD_CLASS_ID, 8 BOX_OUT_OF_IMAGE, 4 BOX_ZERO (các issue có thể trùng ảnh). Đây là số issue của checker pipeline; không dùng tổng các issue để suy ra số ảnh bị loại. Số thực tế 213 lấy từ output quét của đúng phiên train trong notebook.

## 3. Raw có rò rỉ ảnh test không? Mức thổi phồng là bao nhiêu?

Có. 01_intake.csv ghi năm ảnh raw trùng với test/val:

| Ảnh raw | Ảnh test/val trùng |
|---|---|
| img_0049 | test/test_033.jpg |
| img_0101 | test/test_039.jpg |
| img_0167 | test/test_044.jpg |
| img_0174 | test/test_085.jpg |
| img_0221 | test/test_009.jpg |

Nếu các ảnh này thực sự được dùng để train raw, điểm raw có thể lạc quan hơn vì mô hình đã thấy ảnh được dùng trong đánh giá. Tuy nhiên **không thể suy ra số điểm bị thổi phồng từ con số 5**; mAP phụ thuộc đối tượng, nhãn và dự đoán ở từng ảnh. Chênh clean − raw = −0.027 cũng không phải mức ảnh hưởng của rò rỉ, vì clean đồng thời đổi nhãn và loại nhiều ảnh khác.

Muốn đo, tạo thêm biến thể raw bỏ đúng năm ảnh leak, giữ nguyên các phần khác, train cùng model/tham số trên nhiều seed rồi đánh giá trên cùng test/gold labels. So sánh raw đầy đủ với raw không leak là ước lượng thực nghiệm cho thiết lập này. Có thể báo cáo thêm kết quả riêng cho năm ảnh leak và 95 ảnh test còn lại để chẩn đoán. Không dùng test để chọn ngưỡng rồi tiếp tục coi điểm đó là đánh giá độc lập.

## 4. Lớp nào được lợi nhất?

AP50 trung bình và độ lệch chuẩn giữa hai seed:

| Lớp | raw (TB ± SD) | clean (TB ± SD) | clean − raw |
|---|---:|---:|---:|
| car | 0.7375 ± 0.0192 | 0.7407 ± 0.0438 | **+0.0032** |
| person | 0.8221 ± 0.0134 | 0.8239 ± 0.0145 | **+0.0018** |
| truck | 0.6485 ± 0.0127 | 0.5359 ± 0.0732 | **−0.1126** |
| bus | 0.6949 ± 0.0195 | 0.6948 ± 0.0034 | **−0.0001** |

Theo trung bình, **car tăng nhiều nhất**, nhưng chỉ +0.0032; person tăng +0.0018 và bus gần như không đổi. Chưa thể gọi đây là cải thiện rõ ràng với hai seed. Điểm đáng chú ý hơn là truck giảm 0.1126 và clean dao động lớn giữa seed (0.4841 và 0.5876).

Trong 02_error_list_v1, checker báo 18 POSSIBLE_WRONG_CLASS, 53 POSSIBLE_MISSING, 29 POSSIBLE_FALSE_POS, 28 POSSIBLE_LOOSE_BOX; đây là các kiểu sai nhãn có thể tác động AP lớp. V1 cũng ghi 6 BAD_CLASS_ID, 3 BAD_FORMAT, 8 BOX_OUT_OF_IMAGE, 4 BOX_ZERO, 7 DUP_BOX và 6 ảnh MISSING_LABEL_FILE. Nhưng các cảnh báo ngữ nghĩa là gợi ý của model, chưa được người xác nhận; CSV không đủ thông tin để gán từng issue cho lớp hay chứng minh loại lỗi nào gây ra biến động AP. Vì vậy không nên kết luận truck giảm là do một lỗi cụ thể. Cần duyệt nhãn, xem số ground truth theo lớp và confusion matrix trước khi giải thích nguyên nhân.

## 5. Clean còn gì? Queue có vượt ngân sách? Có phải giới hạn trên của dữ liệu sạch?

07_error_list_v3.csv còn 71 cảnh báo, không còn severity error:

| Mã | Số issue |
|---|---:|
| POSSIBLE_FALSE_POS | 21 |
| POSSIBLE_LOOSE_BOX | 20 |
| POSSIBLE_MISSING | 13 |
| POSSIBLE_WRONG_CLASS | 3 |
| BOX_TINY | 7 |
| EMPTY_LABEL | 4 |
| BOX_HUGE | 3 |

Đây là cảnh báo cần con người xem, không phải tất cả đều là lỗi đã xác nhận. Queue có **125 ảnh**, ngân sách là **80**, vượt **45 ảnh**. Pipeline mô phỏng đã xem 80 ảnh; sau đó còn cảnh báo trên **38/203 ảnh (18.72%)**, cao hơn ngưỡng Hold 10%, nên verdict vẫn là **Hold**. Việc mô phỏng review không tương đương người thật đã xác nhận ground truth. Sáu MISSING_LABEL_FILE trong vòng đầu cần người gán nhãn; máy không thể tự tạo nhãn.

Do đó clean_v1 hiện không phải dữ liệu sạch đã chứng nhận, cũng không phải “giới hạn trên” đáng tin cậy của dữ liệu sạch. Đây là bản sau sửa tự động và review mô phỏng có giới hạn. Một bộ sạch/upper bound cần người có chuyên môn xử lý nhãn thiếu, duyệt cảnh báo, xác minh không rò rỉ và xác nhận nhãn. Chưa có bằng chứng đã review đầy đủ để gọi clean_v1 là oracle.

## Kết luận

Raw đạt mAP50 trung bình 0.726 và clean 0.699; với hai seed, chưa thể kết luận về hiệu quả tổng quát của việc làm sạch. Raw có năm ảnh trùng test, nhưng mức điểm bị ảnh hưởng chưa được đo; cần đối chứng raw không leak. JSON ghi 224 ảnh raw được cấp vào train, còn số sau lọc Ultralytics không xác định được nếu thiếu log/cache. Car tăng trung bình ít, truck giảm nhiều nhưng chưa xác định được nguyên nhân theo lớp. Clean còn cảnh báo, queue vượt ngân sách 45 ảnh và verdict Hold; cần hoàn tất review thật trước khi xem đây là dữ liệu sạch.

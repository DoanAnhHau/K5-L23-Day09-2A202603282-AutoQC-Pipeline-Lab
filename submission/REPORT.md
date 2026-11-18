# Báo cáo Day 09 — AutoQC Pipeline Lab

## 1. Kết luận trước

Pipeline nhận **224 ảnh**, loại **21 ảnh** ở intake (3 ảnh quá nhỏ, 2 ảnh hỏng, 8 trùng byte, 3 trùng gần và 5 rò rỉ test/val), còn **203 ảnh** để kiểm nhãn. Sau 80 ảnh được mô phỏng duyệt, gate kết luận **Hold**: không còn lỗi bắt buộc, nhưng còn cảnh báo ở **38/203 ảnh = 18.72%**, cao hơn ngưỡng 10%. Pipeline xuất `clean_v1` dưới dạng **bản nháp Hold**, không phải bản phát hành.

## 2. Lỗi và phân việc

Các số dưới đây là số issue trong `02_error_list_v1` (ảnh, dòng, lớp hoặc dự đoán tùy mã); một ảnh/dòng có thể mang nhiều mã.

| Mã lỗi | Số cờ (ảnh/dòng/lớp/dự đoán) | Ai xử lý | Cách xử lý |
|---|---:|---|---|
| `BAD_SIZE` | 3 | Máy (intake) | Loại ảnh quá nhỏ |
| `CORRUPT_IMAGE` | 2 | Máy (intake) | Loại ảnh hỏng |
| `DUP_IMAGE_EXACT` | 8 | Máy (intake) | Giữ bản đầu, loại bản trùng byte |
| `DUP_IMAGE_NEAR` | 3 | Máy (intake) | Loại bản trùng gần đã kiểm tra |
| `LEAK_TEST` | 5 | Máy (intake) | Loại ảnh trùng kho test/val |
| `BAD_CLASS_ID` | 6 | Máy | Xoá dòng nhãn sai ID |
| `BAD_CLASS_NAME` | 2 tên lớp trong 1 `data.yaml` | Máy | Chuẩn hoá `names` trong `data.yaml` một lần |
| `BAD_FORMAT` | 3 | Máy | Xoá dòng nhãn sai định dạng |
| `BOX_OUT_OF_IMAGE` | 8 | Máy | Clip box về biên ảnh |
| `BOX_ZERO` | 4 | Máy | Xoá dòng box rỗng |
| `DUP_BOX` | 7 | Máy | Xoá dòng box trùng |
| `MISSING_LABEL_FILE` | 6 | Người | Gán nhãn; máy chỉ đưa ảnh lên đầu hàng đợi, không tạo nhãn giả |
| `BOX_HUGE` / `BOX_TINY` | 3 / 6 | Người | Kiểm tra box lớn/nhỏ đáng ngờ |
| `CLASS_RARE` / `EMPTY_LABEL` | 4 lớp / 12 ảnh | Người | Kiểm tra mất cân bằng lớp hoặc ảnh nền/thiếu nhãn |
| `POSSIBLE_FALSE_POS` | 29 | Người | Xem nhãn có thể là box thừa |
| `POSSIBLE_LOOSE_BOX` | 28 | Người | Xem box có lỏng/lệch không |
| `POSSIBLE_MISSING` | 53 | Người | Xem vật model phát hiện nhưng nhãn chưa có |
| `POSSIBLE_WRONG_CLASS` | 18 | Người | Xem nhãn có gán nhầm lớp không |

`03_fixes.csv` ghi **35 hành động**: 20 `drop_line`, 8 `clip_box`, 1 `rename_classes` và 6 `needs_label`. Ở vòng QC đầu có **36 lỗi bắt buộc và 153 cảnh báo** (ngoài 21 lỗi intake). Review queue có **125 ảnh**, vượt ngân sách 80. Queue xếp theo điểm ưu tiên cộng dồn từ loại cảnh báo, mức độ và confidence; ví dụ `img_0020` đứng đầu (11.524) vì có cảnh báo missing/wrong-class cùng một dòng đã sửa, còn 6 ảnh thiếu nhãn có điểm 10 nên cũng được ưu tiên cao. Trong 80 ảnh đầu queue có **94 cảnh báo ngữ nghĩa** (51 missing, 18 wrong-class, 13 false-positive, 12 loose-box) trên 74 ảnh; đây là cảnh báo được đưa ra để xem, không phải 94 lỗi đã được xác nhận.

## 3. Calibration (2 ngưỡng)

Precision/recall dưới đây lấy từ sweep trên `data/calib` (`POSSIBLE_*` theo đúng mã lỗi; thứ tự trong ô là precision/recall). “Trước” là giá trị khởi đầu; “sau” là ngưỡng đã chọn.

| Tham số | Khởi đầu → chọn | Precision / Recall trước → sau | Lý do chọn |
|---|---|---|---|
| `duplicates.phash_dist` | 2 → **8** | Exact: 1.00/1.00 → 1.00/1.00; Near: 1.00/0.50 → 1.00/0.67 | Ở 8, bắt 4/6 near-duplicate với precision 1.00. Ngưỡng 16 bắt 5/6, nhưng trên raw còn gắn nhầm cặp ảnh khác nội dung `img_0008`/`img_0011`; ảnh bị loại là lỗi bắt buộc nên ưu tiên tránh false positive. |
| `second_opinion.conf` | 0.25 → **0.6** | Wrong-class: 0.56/0.83 → 0.62/0.83; Missing: 0.50/1.00 → 0.57/1.00; Loose-box: 0.29/0.40 → 0.20/0.20 | Cảnh báo chỉ đưa vào review nên chấp nhận precision vừa phải để giữ recall. 0.6 cải thiện precision wrong-class và missing mà vẫn giữ recall của hai nhóm; trade-off là loose-box giảm recall. False-positive giữ 0.31/1.00 vì dùng `fp_conf` riêng. |

Calibration có thể gây hiểu nhầm: 39 ảnh và chỉ 6 near-duplicate khiến mỗi ca làm recall đổi khoảng 0.17; ngoài ra bộ calib không có ca `LEAK_TEST` và phân bố lỗi semantic khác raw. Trên raw, ngưỡng 8 gắn cờ 16 ảnh: 8 exact duplicate, 3 near duplicate và 5 leak test. Mở xem cho thấy `img_0076`/`img_0042` và `img_0143`/`img_0037` là cùng ảnh qua xử lý, còn `img_0223`/`img_0077` là cặp gần trùng. Bằng chứng này hỗ trợ ngưỡng 8 trên raw nhưng không chứng minh recall trên dữ liệu khác.

## 4. Cửa chặn và điều pipeline không bắt được

Chạy pipeline với `--budget 0` cho **Rework**, còn **6 lỗi bắt buộc `MISSING_LABEL_FILE`**, exit code **20**; `work/rep_nb/08_summary.json` ghi `exported: null` và không có `work/exp_nb/clean_v1`. Gate không bỏ qua lỗi bắt buộc dù không có người duyệt. Với ngân sách chuẩn 80, người duyệt mô phỏng thay nhãn cho 80 ảnh; vòng cuối còn **0 lỗi bắt buộc**, nhưng **38/203 ảnh** vẫn có cảnh báo nên quyết định là Hold.

Quan sát trực tiếp 10 ảnh `work/viz` (img_0001–0009, img_0030) cho thấy: (1) `img_0001` có hai box person gần trùng quanh người áo vest (dòng 1 và 3), check `DUP_BOX` phát hiện; (2) `img_0002` có người rất nhỏ trên thuyền, khó xác minh box hoặc người bị bỏ sót ở kích thước này; (3) `img_0030` là cảnh đông người, box sát/chồng nhau khiến ranh giới từng người khó kiểm chứng. Hai điểm cuối cần xem người, không khẳng định chắc lỗi chỉ từ hình vẽ nhãn. Điểm mù khác là lỗi ngữ nghĩa khi model và nhãn cùng đồng ý nhưng cùng sai; checker chỉ đưa ý kiến, không phải ground truth. pHash cũng không hiểu nội dung: cặp `img_0008`/`img_0011` cho thấy tăng ngưỡng có thể loại nhầm ảnh tốt.

Để chuyển Hold thành Release, cần người kiểm tra/sửa các cảnh báo còn lại rồi chạy lại đến khi tỷ lệ ảnh còn cảnh báo dưới 10%; mọi lỗi bắt buộc phát sinh sau sửa cũng phải được xử lý.

## 5. Train: raw vs clean (tùy chọn)

Đã train `yolo26n.pt`, 20 epoch, batch 16, imgsz 640, hai seed (0, 1) cho mỗi biến thể. Mỗi biến thể dùng cùng cấu hình và tập đánh giá; raw có 224 ảnh train, clean có 203. Hash config của cả bốn kết quả khớp config hiện tại; hash manifest của hai kết quả clean cũng khớp `clean_v1`.

| Biến thể | Ảnh train | Seed | mAP50 (TB ± SD) | mAP50-95 (TB ± SD) | Precision (TB ± SD) | Recall (TB ± SD) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 2 | 0.699 ± 0.025 | 0.486 ± 0.013 | 0.796 ± 0.050 | 0.596 ± 0.049 |

Chênh mAP50 (clean − raw) là **−0.027**. Độ chênh này cùng cỡ với SD giữa hai seed của clean (0.025), nên với chỉ hai seed và test 100 ảnh chưa thể kết luận clean làm chất lượng tốt hơn; điểm trung bình clean thấp hơn trong lần chạy này. mAP50-95 cũng thấp hơn 0.026; precision tăng 0.032 và recall giảm 0.053.

Các khả năng cần kiểm chứng: (1) raw có 5 ảnh bị báo `LEAK_TEST` ở intake, nên ảnh trùng test có thể làm điểm raw lạc quan hơn; (2) clean chỉ được mô phỏng duyệt 80/125 ảnh trong hàng đợi và vẫn còn cảnh báo ở 38 ảnh, nên nhãn clean chưa được xác nhận hoàn toàn; (3) clean ít hơn raw 21 ảnh và chỉ có hai seed, vì vậy khác biệt có thể do cỡ/phân bố tập train và nhiễu seed. Có thể kiểm bằng đối chiếu hash/near-duplicate giữa train và test, duyệt phần cảnh báo còn lại, rồi chạy thêm seed và so trên cùng tập test.

## 6. Quyết định và ghi chú AI

Quyết định hiện tại là **Hold**, không phát hành `clean_v1` như bản sạch. Cần duyệt thêm các ảnh còn cảnh báo và chạy pipeline lại; mục tiêu là 0 lỗi bắt buộc và tỷ lệ cảnh báo chưa duyệt dưới 10%.

- **Prompt chính:** yêu cầu triển khai các module theo docstring và test, dùng ngưỡng từ config, không sửa dữ liệu raw.
- **Cách kiểm:** đọc diff và các ca biên; chạy test riêng từng module, toàn bộ **79 test**, calibration hai sweep, pipeline thường và gate thử với budget 0; mở trực quan các ảnh/cặp duplicate raw.
- **Lần AI sai/thiếu:** bản autofix ban đầu để ưu tiên drop phụ thuộc thứ tự issue khi một dòng vừa cần clip vừa cần xoá. Khi rà ca biên, đã sửa để `drop_line` luôn thắng; test ca hai lỗi và cả 11 test autofix đều qua.

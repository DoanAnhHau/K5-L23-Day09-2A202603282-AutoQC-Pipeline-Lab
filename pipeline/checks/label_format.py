"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    settings = cfg["label_format"]
    overflow_tol = settings["overflow_tol"]
    min_box_px = settings["min_box_px"]
    max_box_frac = settings["max_box_frac"]
    dup_box_iou = settings["dup_box_iou"]
    issues = []

    for label_file in ds.label_files():
        stem = label_file.stem
        earlier = []  # (class_id, box) từ các dòng hợp lệ phía trước
        image_size = ds.image_size(stem)

        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                class_id, cx, cy, width, height = parse_yolo_line(text)
            except ValueError as exc:
                issues.append(Issue(
                    "label_format", ERROR, "BAD_FORMAT", stem,
                    f"dòng nhãn sai định dạng: {exc}", line=line_no,
                ))
                continue

            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue(
                    "label_format", ERROR, "BAD_CLASS_ID", stem,
                    f"ID lớp {class_id} ngoài khoảng 0..{len(ds.classes) - 1}",
                    line=line_no, extra={"class_id": class_id},
                ))
                continue

            if width <= 0 or height <= 0:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_ZERO", stem,
                    f"chiều rộng và chiều cao box phải dương (w={width}, h={height})",
                    line=line_no,
                ))
                continue

            box = (cx, cy, width, height)
            left, top = cx - width / 2, cy - height / 2
            right, bottom = cx + width / 2, cy + height / 2
            overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
            if overflow > overflow_tol:
                issues.append(Issue(
                    "label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"box tràn ảnh {overflow:.6g} (tolerance {overflow_tol})",
                    line=line_no, extra={"overflow": overflow},
                ))

            if any(prev_class == class_id and iou_xywh(prev_box, box) > dup_box_iou
                   for prev_class, prev_box in earlier):
                issues.append(Issue(
                    "label_format", ERROR, "DUP_BOX", stem,
                    "box trùng với một box cùng lớp ở dòng trước",
                    line=line_no,
                ))

            if image_size is not None:
                image_width, image_height = image_size
                short_side_px = min(width * image_width, height * image_height)
                if short_side_px < min_box_px:
                    issues.append(Issue(
                        "label_format", WARN, "BOX_TINY", stem,
                        f"cạnh ngắn của box {short_side_px:.6g}px < {min_box_px}px",
                        line=line_no, extra={"short_side_px": short_side_px},
                    ))

            if width * height > max_box_frac:
                issues.append(Issue(
                    "label_format", WARN, "BOX_HUGE", stem,
                    f"diện tích box {width * height:.6g} > {max_box_frac}",
                    line=line_no, extra={"area_frac": width * height},
                ))

            earlier.append((class_id, box))

    return issues

"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    removed_stems = set()
    fixes = []

    # Decide image-level removals first so any line issues for those images are ignored.
    for issue in issues:
        if issue.severity != ERROR:
            continue
        if issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes:
            if issue.image in removed_stems:
                continue
            image_paths = [p for p in ds.image_files() if p.stem == issue.image]
            label_path = ds.label_path(issue.image)
            before = ", ".join(p.name for p in image_paths)
            if label_path.exists():
                before = f"{before}, {label_path.name}" if before else label_path.name
            if not image_paths and not label_path.exists():
                continue
            for path in image_paths:
                path.unlink()
            if label_path.exists():
                label_path.unlink()
            removed_stems.add(issue.image)
            fixes.append({
                "image": issue.image, "line": None, "action": "remove_image",
                "before": before, "after": "ảnh và nhãn đã xóa khỏi dataset",
            })

    # Missing labels are deliberately left absent and routed to a human.
    for issue in issues:
        if issue.severity == ERROR and issue.code == "MISSING_LABEL_FILE":
            fixes.append({
                "image": issue.image, "line": None, "action": "needs_label",
                "before": "(không có file)", "after": "chuyển người gán",
            })

    # Gather line-level actions by original line number. Drop always wins over clip.
    line_actions = defaultdict(dict)
    for issue in issues:
        if issue.severity != ERROR or issue.image in removed_stems or issue.line is None:
            continue
        if issue.code in drop_codes:
            line_actions[issue.image][issue.line] = "drop_line"
        elif issue.code == "BOX_OUT_OF_IMAGE":
            line_actions[issue.image].setdefault(issue.line, "clip_box")

    for stem in sorted(line_actions):
        label_path = ds.label_path(stem)
        if not label_path.is_file():
            continue
        original = label_path.read_text(encoding="utf-8", errors="replace")
        rows = original.splitlines()
        changed = False
        for line_no in sorted(line_actions[stem]):
            index = line_no - 1
            if index < 0 or index >= len(rows):
                continue
            action = line_actions[stem][line_no]
            before = rows[index]
            if action == "drop_line":
                rows[index] = None
                after = ""
                changed = True
            else:
                try:
                    class_id, cx, cy, width, height = parse_yolo_line(before)
                except ValueError:
                    continue
                left = max(0.0, cx - width / 2)
                top = max(0.0, cy - height / 2)
                right = min(1.0, cx + width / 2)
                bottom = min(1.0, cy + height / 2)
                clipped_width, clipped_height = right - left, bottom - top
                if clipped_width <= 0 or clipped_height <= 0:
                    rows[index] = None
                    action = "drop_line"
                    after = ""
                else:
                    clipped_cx, clipped_cy = (left + right) / 2, (top + bottom) / 2
                    rows[index] = " ".join(format(value, ".10g") for value in (
                        class_id, clipped_cx, clipped_cy, clipped_width, clipped_height
                    ))
                    after = rows[index]
                changed = True
            fixes.append({
                "image": stem, "line": line_no, "action": action,
                "before": before, "after": after,
            })

        if changed:
            kept_rows = [row for row in rows if row is not None]
            label_path.write_text("\n".join(kept_rows) + ("\n" if original.endswith(("\n", "\r")) else ""), encoding="utf-8")

    # Restore the configured class ordering while preserving other data.yaml keys.
    if any(i.severity == ERROR and i.code == "BAD_CLASS_NAME" for i in issues):
        if ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        before = data.get("names")
        data["names"] = list(cfg["classes"])
        ds.yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        fixes.append({
            "image": "data.yaml", "line": None, "action": "rename_classes",
            "before": repr(before), "after": repr(data["names"]),
        })

    return fixes

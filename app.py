import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "Monitoring APD Konstruksi"
DEFAULT_PPE_MODEL = os.getenv("PPE_MODEL_PATH", "models/best.pt")
DEFAULT_PERSON_MODEL = os.getenv("PERSON_MODEL_PATH", "yolov8n.pt")

REQUIRED_PPE = ["helmet", "mask", "safety vest", "boots", "gloves"]
DISPLAY_NAMES = {
    "helmet": "Helmet",
    "mask": "Mask",
    "safety vest": "Safety Vest",
    "boots": "Boots",
    "gloves": "Gloves",
}
# OpenCV colors are BGR
PPE_COLOR = (255, 190, 0)
PERSON_CHECK_COLOR = (0, 165, 255)
PERSON_COMPLETE_COLOR = (0, 180, 0)
UNASSIGNED_COLOR = (0, 0, 255)


def normalize_class(name):
    name = str(name).strip().lower()
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(name.split())


def canonical_ppe(name):
    name = normalize_class(name)
    aliases = {
        "helmet": "helmet", "helm": "helmet", "hard hat": "helmet",
        "mask": "mask", "face mask": "mask", "facemask": "mask",
        "safety vest": "safety vest", "vest": "safety vest", "rompi": "safety vest",
        "boot": "boots", "boots": "boots", "safety boot": "boots", "safety boots": "boots",
        "glove": "gloves", "gloves": "gloves",
    }
    return aliases.get(name, name)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def point_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def associate_person(ppe_box, ppe_class, people):
    """Heuristik asosiasi berdasarkan titik tengah dan posisi vertikal relatif."""
    x1, y1, x2, y2 = ppe_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ppe_class = canonical_ppe(ppe_class)

    vertical_regions = {
        "helmet": (0.00, 0.38),
        "mask": (0.04, 0.45),
        "safety vest": (0.15, 0.78),
        "gloves": (0.12, 0.92),
        "boots": (0.58, 1.00),
    }
    y_min, y_max = vertical_regions.get(ppe_class, (0.0, 1.0))
    candidates = []

    for idx, person in enumerate(people):
        pbox = person["box"]
        px1, py1, px2, py2 = pbox
        height = max(py2 - py1, 1)
        if not point_inside_box(cx, cy, pbox):
            continue
        relative_y = (cy - py1) / height
        if not (y_min <= relative_y <= y_max):
            continue

        # Prefer the smallest valid person box when people overlap.
        # This remains a heuristic and should be visually validated.
        candidates.append((box_area(pbox), idx))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_models(ppe_model_path, person_model_path):
    if not os.path.isfile(ppe_model_path):
        raise FileNotFoundError(
            f"Bobot model APD tidak ditemukan: {ppe_model_path}. "
            "Letakkan best.pt di folder models/ atau atur PPE_MODEL_PATH."
        )
    ppe_model = YOLO(ppe_model_path)
    person_model = YOLO(person_model_path)
    return ppe_model, person_model


def run_detection(frame_bgr, ppe_model, person_model, ppe_conf, person_conf, imgsz):
    output = frame_bgr.copy()

    ppe_result = ppe_model.predict(
        source=frame_bgr, conf=ppe_conf, imgsz=imgsz, verbose=False
    )[0]
    person_result = person_model.predict(
        source=frame_bgr, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False
    )[0]

    ppe_detections = []
    if ppe_result.boxes is not None:
        for b in ppe_result.boxes:
            coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(b.cls[0].item())
            confidence = float(b.conf[0].item())
            class_name = canonical_ppe(ppe_model.names[class_id])
            ppe_detections.append({
                "box": tuple(coords), "class": class_name, "confidence": confidence
            })

    people = []
    if person_result.boxes is not None:
        for b in person_result.boxes:
            coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
            people.append({
                "box": tuple(coords),
                "confidence": float(b.conf[0].item()),
                "ppe": set(),
                "ppe_conf": {},
            })

    unassigned = []
    for item in ppe_detections:
        idx = associate_person(item["box"], item["class"], people)
        if idx is None:
            unassigned.append(item)
            continue
        person = people[idx]
        cls = item["class"]
        person["ppe"].add(cls)
        person["ppe_conf"][cls] = max(
            item["confidence"], person["ppe_conf"].get(cls, 0.0)
        )

    # Draw PPE boxes and labels.
    for item in ppe_detections:
        x1, y1, x2, y2 = item["box"]
        name = DISPLAY_NAMES.get(item["class"], item["class"])
        label = f"{name} {item['confidence']:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), PPE_COLOR, 2)
        cv2.putText(output, label, (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PPE_COLOR, 2)

    reports = []
    complete_count = 0
    for idx, person in enumerate(people, start=1):
        detected = {canonical_ppe(x) for x in person["ppe"]}
        missing = sorted(set(REQUIRED_PPE) - detected)
        if not missing:
            status = "APD terdeteksi lengkap"
            color = PERSON_COMPLETE_COLOR
            complete_count += 1
        else:
            status = "Perlu verifikasi"
            color = PERSON_CHECK_COLOR

        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"Person {idx}: {status}"
        cv2.putText(output, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        reports.append({
            "Person": idx,
            "Status": status,
            "APD terdeteksi": ", ".join(
                DISPLAY_NAMES.get(x, x) for x in sorted(detected)
            ) if detected else "Belum ada APD terdeteksi",
            "Belum terdeteksi": ", ".join(
                DISPLAY_NAMES.get(x, x) for x in missing
            ) if missing else "-",
            "Confidence person": round(person["confidence"], 3),
            "Confidence APD": {
                DISPLAY_NAMES.get(k, k): round(v, 3)
                for k, v in person["ppe_conf"].items()
            },
        })

    for item in unassigned:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), UNASSIGNED_COLOR, 2)
        cv2.putText(output, "APD tidak terasosiasi", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, UNASSIGNED_COLOR, 2)

    summary = {
        "people_count": len(people),
        "complete_count": complete_count,
        "needs_check": len(people) - complete_count,
        "ppe_counts": {},
        "unassigned_count": len(unassigned),
        "reports": reports,
    }
    for item in ppe_detections:
        label = DISPLAY_NAMES.get(item["class"], item["class"])
        summary["ppe_counts"][label] = summary["ppe_counts"].get(label, 0) + 1

    return output, summary


def show_summary(summary):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manusia terdeteksi", summary["people_count"])
    c2.metric("APD terdeteksi lengkap*", summary["complete_count"])
    c3.metric("Perlu verifikasi", summary["needs_check"])
    c4.metric("APD tidak terasosiasi", summary["unassigned_count"])

    st.caption(
        "*'Lengkap' berarti kelima kelas APD berhasil dideteksi dan diasosiasikan "
        "oleh heuristik ini; bukan jaminan kepatuhan keselamatan."
    )
    st.subheader("Jumlah deteksi APD")
    if summary["ppe_counts"]:
        st.json(summary["ppe_counts"])
    else:
        st.info("Tidak ada deteksi APD di atas threshold.")

    st.subheader("Rincian per orang")
    if summary["reports"]:
        st.dataframe(summary["reports"], use_container_width=True, hide_index=True)
    else:
        st.warning("Tidak ada manusia terdeteksi pada input ini.")


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🦺", layout="wide")
st.title("🦺 Monitoring APD Konstruksi")
st.write(
    "Unggah gambar atau video untuk mendeteksi manusia dan lima kelas APD. "
    "Hasil adalah alat bantu pemeriksaan dan perlu diverifikasi oleh petugas."
)

with st.sidebar:
    st.header("Pengaturan model")
    ppe_model_path = st.text_input("Path model APD", value=DEFAULT_PPE_MODEL)
    person_model_path = st.text_input("Model person", value=DEFAULT_PERSON_MODEL)
    ppe_conf = st.slider("Confidence APD", 0.05, 0.90, 0.25, 0.05)
    person_conf = st.slider("Confidence manusia", 0.05, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Resolusi inferensi", options=[320, 480, 640, 800, 960, 1280], value=640)
    st.caption("Resolusi lebih tinggi dapat membantu objek kecil, tetapi lebih lambat.")

try:
    ppe_model, person_model = load_models(ppe_model_path, person_model_path)
    st.success("Model siap digunakan.")
    with st.expander("Kelas yang dibaca dari model APD"):
        st.write(ppe_model.names)
except Exception as e:
    st.error(str(e))
    st.info(
        "Pastikan file best.pt tersedia di models/best.pt saat dijalankan lokal, "
        "atau atur PPE_MODEL_PATH pada hosting."
    )
    st.stop()

tab_image, tab_video = st.tabs(["📷 Gambar", "🎞️ Video"])

with tab_image:
    uploaded_image = st.file_uploader(
        "Pilih gambar", type=["jpg", "jpeg", "png", "webp"], key="image_upload"
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="Gambar asli", use_container_width=True)

        if st.button("Deteksi APD pada gambar", type="primary", key="detect_image"):
            with st.spinner("Sedang menjalankan deteksi..."):
                annotated, summary = run_detection(
                    frame, ppe_model, person_model, ppe_conf, person_conf, imgsz
                )
            st.subheader("Hasil deteksi")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Gambar hasil deteksi",
                use_container_width=True
            )
            show_summary(summary)

            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button(
                    "Unduh gambar hasil",
                    data=encoded.tobytes(),
                    file_name="hasil_monitoring_apd.jpg",
                    mime="image/jpeg"
                )

with tab_video:
    uploaded_video = st.file_uploader(
        "Pilih video", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
    )
    max_frames = st.number_input(
        "Maksimum frame yang diproses (0 = semua frame)",
        min_value=0, max_value=100000, value=300, step=50
    )
    frame_stride = st.number_input(
        "Proses setiap N frame (1 = semua frame)",
        min_value=1, max_value=60, value=1, step=1
    )
    st.caption(
        "Untuk hosting gratis, mulai dengan video pendek. Video panjang atau resolusi "
        "tinggi dapat melampaui batas waktu/memori hosting."
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Proses video", type="primary", key="detect_video"):
            input_path = None
            output_path = None
            try:
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_video.getbuffer())
                    input_path = tmp.name

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise RuntimeError("Video tidak dapat dibuka. Coba file MP4 yang valid.")

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0 or fps > 120:
                    fps = 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if width <= 0 or height <= 0:
                    raise RuntimeError("Ukuran video tidak valid.")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    output_path = tmp_out.name

                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Gagal membuat video output.")

                progress = st.progress(0.0, text="Memproses video...")
                status_box = st.empty()
                frame_idx = 0
                processed = 0
                latest_summary = None
                stop_after = total_frames if max_frames == 0 else min(total_frames, int(max_frames))

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_idx % int(frame_stride) == 0:
                        annotated, latest_summary = run_detection(
                            frame, ppe_model, person_model,
                            ppe_conf, person_conf, imgsz
                        )
                        processed += 1
                    else:
                        # Reuse previous annotated frame only for display continuity.
                        # This is not a fresh inference for skipped frames.
                        annotated = frame

                    writer.write(annotated)
                    frame_idx += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_idx / max(stop_after, 1), 1.0),
                            text=f"Frame {frame_idx}/{stop_after}"
                        )
                    status_box.write(f"Frame dibaca: {frame_idx} | Frame dianalisis: {processed}")

                    if max_frames != 0 and frame_idx >= int(max_frames):
                        break

                cap.release()
                writer.release()
                progress.empty()

                if processed == 0:
                    raise RuntimeError("Tidak ada frame yang berhasil dianalisis.")

                st.success(f"Video selesai diproses. Frame dianalisis: {processed}.")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    "Unduh video hasil monitoring",
                    data=video_bytes,
                    file_name="hasil_monitoring_apd.mp4",
                    mime="video/mp4"
                )
                if latest_summary:
                    st.subheader("Ringkasan frame terakhir yang dianalisis")
                    show_summary(latest_summary)
                    st.caption(
                        "Ringkasan ini berasal dari frame terakhir yang dianalisis, "
                        "bukan akumulasi unik pekerja sepanjang video."
                    )
            except Exception as e:
                st.error(f"Gagal memproses video: {e}")
            finally:
                for p in [input_path, output_path]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

st.divider()
st.caption(
    "Catatan: asosiasi APD menggunakan heuristik bounding box, belum tracking identitas "
    "atau pose estimation. Jangan gunakan keluaran otomatis sebagai satu-satunya dasar "
    "keputusan keselamatan."
)

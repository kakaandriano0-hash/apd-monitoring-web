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
    """Asosiasikan APD ke orang memakai overlap + zona tubuh + jarak ternormalisasi."""
    x1, y1, x2, y2 = map(float, ppe_box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ppe_class = canonical_ppe(ppe_class)
    zones = {
        "helmet": (0.00, 0.32), "mask": (0.03, 0.42),
        "safety vest": (0.18, 0.78), "gloves": (0.15, 0.92),
        "boots": (0.60, 1.08),
    }
    y_min, y_max = zones.get(ppe_class, (0.0, 1.0))
    ppe_area = max((x2-x1)*(y2-y1), 1.0)
    candidates = []
    for idx, person in enumerate(people):
        px1, py1, px2, py2 = map(float, person["box"])
        pw, ph = max(px2-px1, 1.0), max(py2-py1, 1.0)
        # Allow small box offsets: either center lies inside person or boxes overlap enough.
        ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
        inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
        overlap_ppe = inter / ppe_area
        center_inside = px1 - .04*pw <= cx <= px2 + .04*pw and py1 - .03*ph <= cy <= py2 + .08*ph
        rel_y = (cy-py1)/ph
        zone_ok = y_min <= rel_y <= y_max
        # Score rewards APD box containment and plausible body zone; penalize distant centers.
        norm_dx = abs(cx-(px1+px2)/2) / pw
        score = 2.0*overlap_ppe + (0.45 if center_inside else 0.0) + (0.35 if zone_ok else -0.8) - 0.25*norm_dx
        if (center_inside or overlap_ppe >= 0.25) and zone_ok:
            candidates.append((score, idx))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def reset_tracker(model):
    """Reset tracker antarsesi video jika predictor Ultralytics sudah dibuat."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor is not None else None
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()


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


def run_detection(frame_bgr, ppe_model, person_model, ppe_conf, person_conf, imgsz, track_people=False):
    output = frame_bgr.copy()

    ppe_result = ppe_model.predict(
        source=frame_bgr, conf=ppe_conf, imgsz=imgsz, verbose=False
    )[0]
    if track_people:
        # ByteTrack mempertahankan ID orang antar-frame selama masih dapat dilacak.
        person_result = person_model.track(
            source=frame_bgr, classes=[0], conf=person_conf, imgsz=imgsz,
            tracker="bytetrack.yaml", persist=True, verbose=False
        )[0]
    else:
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
            track_id = None
            if track_people and person_result.boxes.id is not None:
                try:
                    track_id = int(person_result.boxes.id[len(people)].item())
                except (IndexError, TypeError, ValueError):
                    track_id = None
            people.append({
                "box": tuple(coords),
                "confidence": float(b.conf[0].item()),
                "track_id": track_id,
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

        reports.append({
            "Person": (person.get("track_id") if person.get("track_id") is not None else (None if track_people else idx)),
            "Status": status,
            "Jumlah APD terdeteksi": len(detected),
            "Kelengkapan APD (%)": round(100 * len(detected) / len(REQUIRED_PPE)),
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
        "Persentase kelengkapan = jumlah kelas APD berbeda yang terdeteksi / 5 × 100%. "
        "Kelas yang dihitung: Helmet, Mask, Safety Vest, Boots, Gloves. Ini perkiraan "
        "berdasarkan deteksi dan asosiasi bounding box, bukan jaminan kepatuhan keselamatan."
    )
    st.subheader("Jumlah deteksi APD")
    if summary["ppe_counts"]:
        st.dataframe([{"Kelas APD": k, "Jumlah deteksi": v} for k, v in summary["ppe_counts"].items()], use_container_width=True, hide_index=True)
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

# Simpan hasil di session_state agar tidak hilang ketika tombol download memicu rerun Streamlit.
for _key, _default in {
    "video_result_bytes": None,
    "video_report_rows": None,
    "video_summary_meta": None,
    "video_source_name": None,
    "image_result_bytes": None,
    "image_summary": None,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

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
            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                st.session_state["image_result_bytes"] = encoded.tobytes()
                st.session_state["image_summary"] = summary

        # Hasil disimpan pada session_state, jadi tetap tampil saat tombol unduh diklik.
        if st.session_state.get("image_result_bytes") is not None and st.session_state.get("image_summary") is not None:
            st.subheader("Hasil deteksi")
            result_image = Image.open(__import__("io").BytesIO(st.session_state["image_result_bytes"]))
            st.image(result_image, caption="Gambar hasil deteksi", use_container_width=True)
            show_summary(st.session_state["image_summary"])
            st.download_button(
                "Unduh gambar hasil",
                data=st.session_state["image_result_bytes"],
                file_name="hasil_monitoring_apd.jpg",
                mime="image/jpeg",
                key="download_image_result"
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
            # Bersihkan hasil lama hanya ketika pengguna memulai pemrosesan video baru.
            st.session_state["video_result_bytes"] = None
            st.session_state["video_report_rows"] = None
            st.session_state["video_summary_meta"] = None
            st.session_state["video_source_name"] = uploaded_video.name
            input_path = None
            output_path = None
            try:
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_video.getbuffer())
                    input_path = tmp.name

                reset_tracker(person_model)
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
                # Rekap kumulatif per ID tracking sepanjang video.
                cumulative_people = {}
                cumulative_ppe_counts = {}
                cumulative_unassigned = 0
                stop_after = total_frames if max_frames == 0 else min(total_frames, int(max_frames))

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_idx % int(frame_stride) == 0:
                        annotated, latest_summary = run_detection(
                            frame, ppe_model, person_model,
                            ppe_conf, person_conf, imgsz, track_people=True
                        )
                        processed += 1
                        cumulative_unassigned += latest_summary.get("unassigned_count", 0)
                        for label, count in latest_summary.get("ppe_counts", {}).items():
                            cumulative_ppe_counts[label] = max(cumulative_ppe_counts.get(label, 0), count)
                        for report in latest_summary.get("reports", []):
                            person_id = report.get("Person")
                            # Tracking IDs memungkinkan orang yang sama digabung antar-frame.
                            if person_id is None:
                                continue
                            rec = cumulative_people.setdefault(person_id, {
                                "Person": person_id, "seen_frames": 0,
                                "ppe_classes": set(), "class_hits": {cls: 0 for cls in REQUIRED_PPE},
                                "confidence_person": 0.0, "ppe_conf": {}
                            })
                            rec["seen_frames"] += 1
                            rec["confidence_person"] = max(rec["confidence_person"], report.get("Confidence person", 0.0))
                            for cls in REQUIRED_PPE:
                                display = DISPLAY_NAMES[cls]
                                if display in report.get("APD terdeteksi", ""):
                                    rec["ppe_classes"].add(cls)
                                    rec["class_hits"][cls] += 1
                            for cls, conf in report.get("Confidence APD", {}).items():
                                rec["ppe_conf"][cls] = max(rec["ppe_conf"].get(cls, 0.0), conf)
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

                # Simpan bytes dan laporan ke session_state sebelum file temporer dihapus.
                with open(output_path, "rb") as f:
                    st.session_state["video_result_bytes"] = f.read()
                cumulative_reports = []
                complete_count = 0
                for person_id, rec in sorted(cumulative_people.items(), key=lambda x: x[0]):
                    detected = rec["ppe_classes"]
                    missing = set(REQUIRED_PPE) - detected
                    percent = round(100 * len(detected) / len(REQUIRED_PPE))
                    if not missing:
                        complete_count += 1
                    cumulative_reports.append({
                        "Person ID": person_id,
                        "Status deteksi": "5/5 kelas pernah terdeteksi" if not missing else "Belum semua kelas terdeteksi",
                        "Jumlah APD terdeteksi": f"{len(detected)}/5",
                        "Kelengkapan APD (%)": percent,
                        **{DISPLAY_NAMES[x]: ("Terdeteksi" if x in detected else "Belum terdeteksi") for x in REQUIRED_PPE},
                        "APD pernah terdeteksi": ", ".join(DISPLAY_NAMES[x] for x in REQUIRED_PPE if x in detected) or "Belum ada",
                        "Belum terdeteksi sepanjang video": ", ".join(DISPLAY_NAMES[x] for x in REQUIRED_PPE if x in missing) or "-",
                        "Frame terlihat": rec["seen_frames"],
                    })
                st.session_state["video_report_rows"] = cumulative_reports
                st.session_state["video_summary_meta"] = {
                    "people_count": len(cumulative_people),
                    "complete_count": complete_count,
                    "processed_frames": processed,
                }
                st.success(f"Video selesai diproses. Frame dianalisis: {processed}.")
            except Exception as e:
                st.error(f"Gagal memproses video: {e}")
            finally:
                for p in [input_path, output_path]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        # Render dari session_state setiap rerun, termasuk rerun yang terjadi saat tombol unduh diklik.
        if st.session_state.get("video_result_bytes") is not None:
            st.subheader("Hasil video")
            st.video(st.session_state["video_result_bytes"])
            st.download_button(
                "Unduh video hasil monitoring",
                data=st.session_state["video_result_bytes"],
                file_name="hasil_monitoring_apd.mp4",
                mime="video/mp4",
                key="download_video_result"
            )
            meta = st.session_state.get("video_summary_meta") or {}
            rows = st.session_state.get("video_report_rows") or []
            st.subheader("Rekap kumulatif sepanjang video")
            m1, m2, m3 = st.columns(3)
            m1.metric("Orang unik terlacak", meta.get("people_count", 0))
            m2.metric("Orang dengan 5/5 APD", meta.get("complete_count", 0))
            m3.metric("Frame dianalisis", meta.get("processed_frames", 0))
            st.caption(
                "Orang unik dihitung menggunakan ID ByteTrack. Persentase APD adalah gabungan "
                "kelas APD yang pernah terdeteksi pada orang tersebut di frame yang dianalisis; "
                "bukan bukti APD dipakai dengan benar setiap saat. ID dapat berubah saat pekerja tertutup/berpapasan."
            )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                import pandas as pd
                csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Unduh laporan per orang (CSV)", data=csv_bytes,
                    file_name="laporan_monitoring_apd.csv", mime="text/csv",
                    key="download_video_report_csv"
                )
            else:
                st.info("Tidak ada orang yang berhasil dilacak pada video ini.")
            st.caption(
                "Laporan ditampilkan terpisah di halaman web. Kotak pada video hanya menunjukkan lokasi objek. "
                "Persentase = jumlah kelas APD berbeda yang pernah terdeteksi dibagi 5."
            )

st.divider()
st.caption(
    "Catatan: sistem memakai deteksi orang + ByteTrack dan asosiasi APD berbasis overlap/zona tubuh. "
    "Hasil tetap perlu diverifikasi petugas; untuk akurasi lebih tinggi, evaluasi best.pt pada "
    "video CCTV lokasi sebenarnya dan tambahkan data latih untuk objek kecil/tertutup."
)

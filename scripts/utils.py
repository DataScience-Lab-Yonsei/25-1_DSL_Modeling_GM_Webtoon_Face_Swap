import os
import cv2
import json
import time
import glob
import requests
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import shutil

class Do_OCR_FACECROP():
    PROTAGONIST_LABEL = 1

    def __init__(self, episode_num, ocr_api_key, replacement_word, font_path,
                 target_words=["신재현", "재현", "신팀장", "신선생"],
                 margin=5, inpaint_radius=5, model_path="model_weights.pth", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg19(pretrained=False)
        num_features = self.model.classifier[6].in_features
        num_classes = 2  # 실제 분류 클래스 수 (필요시 변경)
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.episode = str(episode_num)
        self.ocr_api_key = ocr_api_key
        self.replacement_word = replacement_word
        self.target_words = target_words
        self.font_path = font_path
        self.margin = margin
        self.inpaint_radius = inpaint_radius

        # dataset/<episode>/img, label
        self.img_dir = f"dataset/{self.episode}/img"
        self.result_dir = f"results/{self.episode}/ocr_results"
        os.makedirs(self.result_dir, exist_ok=True)

        self.ocr_results = {}

        # 1) OCR
        self.ocr()
        # 2) Inpainting + 텍스트 치환
        self.inpaint_and_replace()
        # 3) 얼굴 크롭 (해당 episode_num만)
        self.naive_face_crop(episode_num)
        # 4) 주인공 여부 분류
        self.protagnoist_face_list = self.is_it_protagonist()
        # 5) 주인공 얼굴만 최종 cropped_faces / cropped_faces_json 에 복사
        self.get_protagonist_face_json(face_list=self.protagnoist_face_list)

        # [중요] temporary 폴더 삭제하지 않도록 아래 줄 주석 처리 또는 제거
        # self.remove_temporary_dirs(dirs=["temporary_crops", "temporary_jsons"])

    def run_ocr_api(self, image_path):
        """ OCR API 호출 """
        url = "https://api.upstage.ai/v1/document-ai/ocr"
        headers = {"Authorization": f"Bearer {self.ocr_api_key}"}
        with open(image_path, "rb") as f:
            files = {"document": f}
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()

    def get_text_color(self, image_path, bbox, threshold=200):
        """ OCR 영역의 대표 색상(검정색 가정)을 추출하기 위한 예시 메서드 """
        image = cv2.imread(image_path)
        if image is None or len(bbox) != 4:
            return {"r": 0, "g": 0, "b": 0}
        x_min = min(pt["x"] for pt in bbox)
        y_min = min(pt["y"] for pt in bbox)
        x_max = max(pt["x"] for pt in bbox)
        y_max = max(pt["y"] for pt in bbox)
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return {"r": 0, "g": 0, "b": 0}
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = roi.reshape(-1, 3)
        filtered = [tuple(c) for c in pixels if np.mean(c) < threshold]
        return dict(zip(["r", "g", "b"], filtered[0])) if filtered else {"r": 0, "g": 0, "b": 0}

    def get_text_size(self, bbox):
        """ OCR bbox의 폭, 높이 계산 """
        if len(bbox) != 4:
            return {"width": 0, "height": 0}
        width = max(pt["x"] for pt in bbox) - min(pt["x"] for pt in bbox)
        height = max(pt["y"] for pt in bbox) - min(pt["y"] for pt in bbox)
        return {"width": width, "height": height}

    def ocr(self):
        """ 이미지별로 OCR 수행 후, 타겟 단어(예: '신재현' 등) 포함된 영역만 추출 """
        img_paths = sorted([p for ext in ["*.jpg", "*.png"] for p in glob.glob(f"{self.img_dir}/{ext}")])
        for path in img_paths:
            try:
                result = self.run_ocr_api(path)
                matched_words = []
                for word in result.get("pages", [{}])[0].get("words", []):
                    text = word.get("text", "")
                    if any(t in text for t in self.target_words):
                        bbox = word.get("boundingBox", {}).get("vertices", [])
                        color = self.get_text_color(path, bbox)
                        size = self.get_text_size(bbox)
                        matched_words.append({
                            "text": text, "bbox": bbox, "color": color, "size": size
                        })
                self.ocr_results[path] = matched_words
                print(f" OCR success: {path}")
                time.sleep(1)
            except Exception as e:
                print(f" OCR failed: {path} ({e})")

    def inpaint_and_replace(self):
        """ OCR 영역을 마스크로 만들어 테두리 보정(inpaint) 후, replacement_word로 치환하여 저장 """
        for img_path, word_infos in self.ocr_results.items():
            original = cv2.imread(img_path)
            if original is None:
                continue
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            for word in word_infos:
                bbox = word["bbox"]
                if len(bbox) != 4:
                    continue
                x_min = max(min(pt["x"] for pt in bbox) - self.margin, 0)
                y_min = max(min(pt["y"] for pt in bbox) - self.margin, 0)
                x_max = min(max(pt["x"] for pt in bbox) + self.margin, original.shape[1])
                y_max = min(max(pt["y"] for pt in bbox) + self.margin, original.shape[0])
                mask[y_min:y_max, x_min:x_max] = 255

            # inpaint
            inpainted = cv2.inpaint(original, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
            pil_img = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 텍스트 다시 쓰기
            for word in word_infos:
                bbox = word["bbox"]
                color = word["color"]
                size = word["size"]
                if len(bbox) != 4:
                    continue
                font_size = max(int(size["height"] * 1.15), 10)
                font = ImageFont.truetype(self.font_path, font_size)
                x_min = min(pt["x"] for pt in bbox)
                y_min = min(pt["y"] for pt in bbox)
                text_color = (color["r"], color["g"], color["b"])
                draw.text((x_min, y_min), self.replacement_word, fill=text_color, font=font)

            save_path = os.path.join(self.result_dir, os.path.basename(img_path))
            pil_img.save(save_path)
            print(f" Saved: {save_path}")

    def naive_face_crop(self, episode_num):
        """
        주어진 에피소드에 대해,
        dataset/<episode>/label 내 JSON(얼굴좌표) 정보를 참고하여 
        얼굴을 크롭한 뒤 results/<episode>/temporary_crops, temporary_jsons 폴더에 저장.
        """

        episode_path = f"dataset/{episode_num}"
        img_folder = f"{episode_path}/img"
        label_folder = f"{episode_path}/label"

        
        output_episode_dir_img = f"results/{episode_num}/temporary_crops"
        os.makedirs(output_episode_dir_img, exist_ok=True)

        output_episode_dir_json = f"results/{episode_num}/temporary_jsons"
        os.makedirs(output_episode_dir_json, exist_ok=True)

        faces_failed = []  # 얼굴 검출 실패 이미지 리스트

        
        if not os.path.isdir(episode_path):
            print(f"[naive_face_crop] Episode directory not found: {episode_path}")
            return


        for img_file in os.listdir(img_folder):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = f"{img_folder}/{img_file}"
            base_name, ext = os.path.splitext(img_file)
            json_path = f"{label_folder}/{base_name}.json"

            if not os.path.exists(json_path):
                faces_failed.append(img_file)
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    faces_failed.append(img_file)
                    continue

            faces = data.get("faces", [])
            if not faces:
                faces_failed.append(img_file)
                continue

            image = Image.open(img_path)

            for idx, face in enumerate(faces, start=1):
                landmark = face.get("landmark")
                if not landmark:
                    continue

                # 윤곽 부위 키
                contour_keys = [f"contour_left{i}" for i in range(1, 10)] + \
                               [f"contour_right{i}" for i in range(1, 10)] + \
                               ["contour_chin"]

                contour_points = []
                for key in contour_keys:
                    point = landmark.get(key)
                    if point:
                        contour_points.append((point["x"], point["y"]))

                if not contour_points:
                    faces_failed.append(img_file)
                    continue

                xs, ys = zip(*contour_points)
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)

                # 패딩 (이미지 크롭)
                padding_x = int((max_x - min_x) * 0.2)
                padding_y_top = int((max_y - min_y) * 1.0)
                padding_y_bottom = int((max_y - min_y) * 0.05)

                eyebrow_y = min(
                    landmark.get("left_eyebrow_upper_middle", {"y": min_y})["y"],
                    landmark.get("right_eyebrow_upper_middle", {"y": min_y})["y"]
                )
                extra_space_above_eyebrow = int((max_y - min_y) * 0.4)
                limited_top = max(0, eyebrow_y - extra_space_above_eyebrow)

                new_left = max(0, min_x - padding_x)
                new_top = max(0, min_y - padding_y_top)
                new_top = min(new_top, limited_top)
                new_right = min(image.width, max_x + padding_x)
                new_bottom = min(image.height, max_y + padding_y_bottom)

                if new_right <= new_left or new_bottom <= new_top:
                    continue

                # 얼굴 크롭
                face_crop = image.crop((new_left, new_top, new_right, new_bottom))
                output_filename = f"{base_name}_face{idx}{ext}"
                output_path = f"{output_episode_dir_img}/{output_filename}"
                face_crop.save(output_path)

                # 크롭 정보 JSON 저장
                json_output_filename = f"{base_name}_face{idx}.json"
                json_output_path = f"{output_episode_dir_json}/{json_output_filename}"
                face_crop_info = {
                    "episode": episode_num,
                    "original_image": img_file,
                    "face_index": idx,
                    "crop_box": {
                        "left": new_left,
                        "top": new_top,
                        "right": new_right,
                        "bottom": new_bottom
                    }
                }
                with open(json_output_path, "w", encoding="utf-8") as jf:
                    json.dump(face_crop_info, jf, indent=4, ensure_ascii=False)

        # ──────────────────────────────────────────────────────
        # 저화질 이미지 필터링
        file_size_threshold = 7.21  # KB
        min_resolution = 100        # 픽셀

        for file in os.listdir(output_episode_dir_img):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = f"{output_episode_dir_img}/{file}"
                file_size_kb = os.path.getsize(file_path) / 1024.0
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                except Exception:
                    os.remove(file_path)
                    continue

                if file_size_kb <= file_size_threshold or width < min_resolution or height < min_resolution:
                    os.remove(file_path)
        # ──────────────────────────────────────────────────────

        # 실패 목록 저장
        if faces_failed:
            failed_file_path = f"results/{episode_num}/failed_faces.txt"
            with open(failed_file_path, "w", encoding="utf-8") as f:
                for fname in faces_failed:
                    f.write(fname + "\n")

    def is_it_protagonist(self):
        """
        results/<episode>/temporary_crops 폴더 안의 얼굴들을 
        self.model로 분류한 뒤, 주인공 클래스(PROTAGONIST_LABEL)면 리스트에 추가.
        """
        temp_crops_dir = f"results/{self.episode}/temporary_crops"


        # temporary_crops 폴더가 없는 경우 예외 처리
        if not os.path.exists(temp_crops_dir):
            print(f"Cannot find temporary_crops: {temp_crops_dir}")
            return []

        cropped_face_list = os.listdir(temp_crops_dir)
        protagonist_face_list = []

        for face in cropped_face_list:
            face_path = f"{temp_crops_dir}/{face}"
            if not os.path.isfile(face_path):
                continue

            img = Image.open(face_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                pred = torch.argmax(output, dim=1).item()

            if pred == self.PROTAGONIST_LABEL:
                protagonist_face_list.append(face)

        return protagonist_face_list

    def get_protagonist_face_json(self, face_list: list):
        """
        주인공으로 분류된 얼굴 이미지를
        cropped_faces, cropped_faces_json 폴더로 복사
        """
        temp_crops_dir = f"results/{self.episode}/temporary_crops"
        temp_jsons_dir = f"results/{self.episode}/temporary_jsons"
        final_crops_dir = f"results/{self.episode}/cropped_faces"
        final_json_dir = f"results/{self.episode}/cropped_faces_json"

        os.makedirs(final_crops_dir, exist_ok=True)
        os.makedirs(final_json_dir, exist_ok=True)

        for face in face_list:
            base_name = os.path.splitext(face)[0]

            temp_face_path = f"{temp_crops_dir}/{face}"
            temp_json_path = f"{temp_jsons_dir}/{base_name}.json"

            final_face_path = f"{final_crops_dir}/{face}"
            final_json_path = f"{final_json_dir}/{base_name}.json"

            if os.path.exists(temp_face_path):
                shutil.copy2(temp_face_path, final_face_path)
            if os.path.exists(temp_json_path):
                shutil.copy2(temp_json_path, final_json_path)




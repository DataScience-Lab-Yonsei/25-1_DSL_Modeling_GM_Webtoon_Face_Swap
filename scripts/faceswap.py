import requests
import os
import base64

class FaceSwapper:
    """
    FaceSwapper:
      - 프로젝트 내 scripts 폴더에 존재한다고 가정.
      - 기본적으로 folder_number=112(예: results/112/...) 폴더를 사용.
      - 생성자에서 project 루트까지의 경로를 찾아내어,
        style_transferred_images, real_faces, face_swapped_images 폴더를 자동으로 할당함.
    """

    def __init__(self, 
                 api_key: str, 
                 folder_number: int = 112, 
                 url: str = "https://api.segmind.com/v1/faceswap-v3"):
        self.api_key = api_key
        self.folder_number = folder_number
        self.url = url
        
        # 현재 faceswap.py 파일이 있는 scripts 디렉토리 경로
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # project 루트 디렉토리 (scripts 폴더의 상위)
        project_root = os.path.dirname(script_dir)

        # 상대 경로 설정
        self.webtoon_dataset_path = os.path.join(
            project_root, "results", str(self.folder_number), "style_transferred_images"
        )
        self.real_dataset_path = os.path.join(project_root, "real_faces")
        self.output_dir = os.path.join(
            project_root, "results", str(self.folder_number), "face_swapped_images"
        )

        # 결과물이 저장될 기본 폴더 생성
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def image_file_to_base64(image_path: str) -> str:
        """이미지 파일을 base64 문자열로 변환"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def swap_faces(self) -> None:
        """
        webtoon_dataset_path의 웹툰 얼굴 이미지를
        real_dataset_path의 실제 얼굴 이미지와 합성 후
        output_dir에 저장한다.
        """
        # 폴더 내 파일 목록(불필요한 파일 제외)
        webtoon_files = [f for f in os.listdir(self.webtoon_dataset_path) 
                         if f != ".DS_Store" and not f.startswith("_")]
        real_files = [f for f in os.listdir(self.real_dataset_path) 
                      if f != ".DS_Store" and not f.startswith("_")]

        for real_face in real_files:
            # 실제 얼굴 파일명에서 확장자를 뺀 이름
            face_jooin = os.path.splitext(real_face)[0]
            target_path = os.path.join(self.real_dataset_path, real_face)

            # 실제 얼굴별로 결과 저장 폴더를 생성
            result_dir = os.path.join(self.output_dir, face_jooin)
            os.makedirs(result_dir, exist_ok=True)

            for webtoon_face in webtoon_files:
                # 웹툰 얼굴 파일명에서 확장자를 뺀 이름
                file_name = os.path.splitext(webtoon_face)[0]
                source_path = os.path.join(self.webtoon_dataset_path, webtoon_face)

                # API에 전송할 JSON 데이터
                data = {
                    "source_img": self.image_file_to_base64(target_path),
                    "target_img": self.image_file_to_base64(source_path),
                    "input_faces_index": 0,
                    "source_faces_index": 0,
                    "face_restore": "codeformer-v0.1.0.pth",
                    "image_quality": 100,
                    "base64": False
                }

                # API 요청 헤더
                headers = {
                    'x-api-key': self.api_key
                }

                # API 요청
                response = requests.post(self.url, json=data, headers=headers)

                # 최종 결과물 이미지 경로
                final_path = os.path.join(result_dir, f"{file_name}_fs.jpg")

                # 이미지 저장
                with open(final_path, "wb") as f:
                    f.write(response.content)


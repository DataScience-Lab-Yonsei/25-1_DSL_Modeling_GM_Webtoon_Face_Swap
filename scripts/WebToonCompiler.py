from PIL import Image
from tqdm.notebook import tqdm
import os
import shutil
import requests
import json
import numpy as np


from bs4 import BeautifulSoup as bs
import requests
import os


Image.MAX_IMAGE_PIXELS = None

class DataJoonbiJoonbi():
    def __init__(self, title : str, api_key : str, api_secret : str, project_direc : str, skip_download : bool, skip_preprocess : bool):

        self.project_driec = project_direc
        self.dataset_direc = os.path.join(self.project_driec, "dataset")
        self.rawdata_direc = os.path.join(self.project_driec, "rawdata")

        self.title = title

        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"

        if not skip_download:
            self.download_image()

        if not skip_preprocess:
            self.ThisWillDoTheJob()
            print("You're all set!")


        if skip_download and skip_preprocess:
            print(">>>> !! Automated Downloading & Preprocessing are both OFF !! <<<<")    

    def get_proprocessed_direc(self, episode_num : int) -> tuple:

        """ 
        특정 에피소드의 전처리된 디렉토리를 (이미지_디렉토리, json_토리) 형태의 튜플로 반환합니다
        """
        
        episode_img_direc = os.path.join(self.dataset_direc, episode_num, "img")
        episode_json_direc = os.path.join(self.dataset_direc, episode_num, "label")

        return (episode_img_direc, episode_json_direc)

    def download_image(self) -> None:
        """ 
        rawdata 디렉토리에 모든 에피소드의 이미지를 저장합니다
        """

        headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

        # Enter WebToon Title
        title = self.title # 띄어쓰기, 맞춤법 지키셔야합니다! (그냥 웹툰 페이지에서 복붙하셔야..ㅎㅎ)

        if not os.path.isdir(self.rawdata_direc):
            os.mkdir(self.rawdata_direc)


        # 네이버 웹툰 검색창에서 검색결과 긁어오기
        search_url = "https://m.comic.naver.com/search/result?keyword=" + title.replace(" ", "%20")
        search_page = requests.get(search_url, headers=headers) # headers 파라미터에 User-Agent 딕셔너리를 패스하고, url을 전달하여 내용물을 가져옵니다.

        # 200이 나와야 정상적으로 서버와의 소통이 완료된 것입니다!
        print(f"Status : {search_page.status_code}")

        search_page = search_page.text # reauests.get(url)로 가져온 내용물 중 text만을 가져와야 html parsing이 가능합니다. 이후 단계에서는 설명을 생략합니다.
        search_page = bs(search_page, "html.parser") # BeautifulSoup 라이브러리를 이용하여 html 형식에 맞게 parsing 해줍니다.

        # # html document
        # print("-"*50 + "아래는 html 문서입니다." + "-"*50,search_page)


        # 웹툰 검색창에서 기본 정보 회수하기
        toon = {}

        # 검색한 웹툰이 맞는지 확인
        title_ = str(search_page.find_all("strong", attrs={"class" : None})[0])[8:-9] # find_all() 함수는 "strng" 이라는 모든 태그들 중, 
                                                                                    # "class" attribute이 없는 즉 'None'인 태그를 찾아 리스트로 반환합니다.

        if title == title_:
            toon["title"] = title_ # 웹툰 제목이 맞는 경우 웹툰 제목 추가
        else:
            print("Recheck Webtoon Title")

        # 에피소드 리스트의 url 회수
        toon_lists_url = (search_page.find_all("div", attrs={"class" : "lst"}))[0].find("a").get("href") # find_all()로 찾은 <a> tag attribute 중 href의 정보를 회수합니다.
        toon_lists_url = "https://m.comic.naver.com/" + toon_lists_url
        toon["toon_lists_url"] = toon_lists_url

        # 네이버 웹툰 상에 등록되어있는 웹툰의 아이디 회수
        toon_id = toon_lists_url.split("=")[-1]
        toon["toon_id"] = int(toon_id)

        # # 현재까지 회수한 기본 정보를 확인합니다
        # for attr, info in toon.items():
        #     print(f"WebTon {attr} : {info}")

        # 에피소드 리스트가 나와있는 페이지의 html을 가져옵니다
        toon_list = bs(requests.get(toon["toon_lists_url"], headers=headers).text, "html.parser")

        # 에피소드 리스트에 대한 정보 회수

        page_num= toon_list.find_all("div", attrs={"class" : "paging_type2"})[0] 
        curr_url_ = page_num.find_all("a", attrs = {"onclick" : "nclk_v2(event,'lst.next')"})[0].get("href")
        curr_url_ = "https://m.comic.naver.com" + curr_url_[:-1] # -> 현재 리스트 페이지의 주소를 회수합니다. 가장 마지막 숫자가 페이지 수를 나타내므로 제거합니다.

        page_num = page_num.find('em', class_='current_pg') 
        page_num = int(page_num.find('span', class_='total').string) # -> 가장 하단의 페이지 수 정보를 가져옵니다

        toon["page_num"] = page_num # 힘들게 얻어냈으니 잘 저장해줍니다.

        for curr_page in range(page_num, 0, -1): # 앞서 얻은 페이지 수 정보와 리스트 주소 정보를 바탕으로 모든 페이지의 모든 에피소드를 긁어오겠습니다.
            # 현재 리스트의 html을 가져옵니다
            curr_url = f"{curr_url_}{curr_page}"
            curr_list = bs(requests.get(curr_url, headers=headers).text, "html.parser")

            # 개별 에피소드 정보에 접근합니다                           
            episodes = curr_list.find_all("li", attrs= {"class" : "item", "data-no": not None, "data-free-convert-date": None})
                                                                                            # 유료분 회차는 로그인이 필요하므로 회피합니다.
            # 현재 리스트의 모든 에피소드에 대해 순회합니다
            for episode in episodes:
                # 개별 에피소드 주소 회수
                episode_link = episode.find_all("a")[0].get('href')
                episode_link = "https://m.comic.naver.com"+episode_link

                # 개별 에피소드 제목 회수
                episode_title = episode.find_all("span", attrs={"class" : "name"})[0]
                episode_title = str(episode_title.contents[0])[8:-9]

                # if "112" not in episode_title:
                #     continue

                # 개별 에피소드의 html을 가져옵니다
                viewing_page = bs( requests.get(episode_link, headers=headers).text, "html.parser" )

                # 개별 에피소드의 모든 이미지 태그를 가져옵니다
                images = viewing_page.find_all("div", attrs={"class" : "toon_view_lst", "id" : "toonLayer"})[0]
                images = images.find_all("img")

                # 앞서 회수한 모든 이미지 태그에서 이미지 url을 추출합니다
                image_urls ={} # 여기에 모든 이미지 url을 저장합니다
                for idx, image in enumerate(images):
                    try: # 간혹 잘못된 이미지 태그가 긁혀오는 경우가 있어 예외처리를 진행해줍니다
                        image_urls[idx] = image["data-src"]
                        
                    except KeyError as e:
                        pass

                # 에피소드 제목을 이름으로한 디렉토리를 만들어줍니다
                episode_direc = os.path.join(self.rawdata_direc, episode_title)

                if not os.path.isdir(episode_direc):
                    os.mkdir(episode_direc)

                # 각 에피소드의 모든 이미지를 다운로드하여, 지정된 경로에 저장합니다
                for idx, url in image_urls.items():
                    img = requests.get(url, headers=headers).content
                    img_path = os.path.join(episode_direc, f"{idx}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(img)
                print(f"Episode {episode_title} is saved!")

    def ThisWillDoTheJob(self) -> None:

        if not os.path.isdir(self.dataset_direc):
            os.mkdir(self.dataset_direc)

        ep_list = [d for d in os.listdir(self.rawdata_direc) if os.path.isdir(os.path.join(self.rawdata_direc, d))]
        ep_num_list = [ i.split()[0][:-1] for i in ep_list]

        for episode_num, epidose_direc in tqdm(zip(ep_num_list, ep_list), total=len(ep_num_list)):
            # if episode_num != "112" :
            #     continue
            dataset_ep_direc = os.path.join(self.dataset_direc, episode_num)
            split_img_direc = os.path.join(dataset_ep_direc, "img")
            json_label_direc = os.path.join(dataset_ep_direc, "label")

            if not os.path.isdir(dataset_ep_direc):
                os.mkdir(dataset_ep_direc)
            if not os.path.isdir(split_img_direc):
                os.mkdir(split_img_direc)

            img_files_direc = os.path.join(self.rawdata_direc, epidose_direc)
            compiled_img = self.compile_images(img_files_direc)
            self.dissect_image(compiled_img, split_img_direc)
            # print(split_img_direc)

            if not os.path.isdir(json_label_direc):
                os.mkdir(json_label_direc)

            img_list = os.listdir(split_img_direc)

            for img_file_name in img_list:

                image_path = os.path.join(split_img_direc, img_file_name)
                img = open(image_path, 'rb')

                if self.is_it_blank(img_path=image_path): # 저희 데이터셋에 빈칸인 컷 많습니다. 얘네까지 api 요청하고 앉아있으니 시간이 너무 오래걸려서 추가합니다.
                    
                    continue
                
                json_path = os.path.join(json_label_direc, f"{img_file_name[:-4]}.js")
                self.get_json(img, json_path)
                    

            print(f"Generating labels for episode_{episode_num} is completed!")

    def get_json(self, img : Image, json_path : str) -> None:

        files = {'image_file': img}
        data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'return_landmark': '1',
                'return_attributes': 'gender,age,smiling,headpose,facequality,blur,eyestatus,emotion'
                }

        response = requests.post(self.api_url, files=files, data=data)

        try:
            json_label = response.json()
        except json.JSONDecodeError:
            print("Failed to decode JSON. Response content:", response.text)
            return

        with open(json_path, 'w') as file:
            json.dump(json_label, file, indent=4)

    def is_it_blank(self, img_path : str) -> bool:

        image = Image.open(img_path)
        # Get pixel data
        pixel_data = list(image.getdata())
        
        pixel_data = np.array(pixel_data)

        is_blank = (np.std(pixel_data) < 2)
        
        return is_blank
        
    def compile_images(self, image_folder : str) -> Image :
        # Get list of image files in the folder
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        # if ".DS_Store" in image_files:
        #     image_files.remove(".DS_Store")
        
        # Sort image files to maintain order
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
        
        # Open images and get their sizes
        images = [Image.open(os.path.join(image_folder, img)) for img in image_files]
        widths, heights = zip(*(img.size for img in images))
        
        # Calculate total height and max width
        total_height = sum(heights)
        max_width = max(widths)
        
        # Create a new blank image with the calculated dimensions
        compiled_image = Image.new('RGB', (max_width, total_height))
        
        # Paste images one below the other
        y_offset = 0
        for img in images:
            compiled_image.paste(img, (0, y_offset))
            y_offset += img.height
        
        return compiled_image

    def dissect_image(self, input_image: Image, output_folder: str) -> None:
        # Open the compiled image
        img = input_image
        
        # Convert image to grayscale
        gray_img = img.convert('L')
        total_height = gray_img.height
        
        # Get pixel data
        pixels = gray_img.load()
        
        # Initialize variables
        cuts = []
        start_y = 0
        curr = None
        prev = None
        
        # Iterate over each row to find cuts
        for y in range(total_height):
            row_pixels = [pixels[x, y] for x in range(gray_img.width)]
            is_pixel_monotone = all(p == 255 or p == 0 for p in row_pixels)
            prev = curr
            curr = is_pixel_monotone

            if y == 0 : # 가장 윗 부분 누락 방지
                start_y = y

            elif y == total_height-1: # 가장 아랫 부분 누락 방지
                end_y = y

                cuts.append((start_y, end_y))

            else:
                if curr == prev: # 이전의 상황이 지속되는 상태
                    continue
                else: # 내용이 있는 부분이 끝났거나 (False -> True), 같은 픽셀값으로 채워진 영역이 끝났거나 (True -> False)
                    end_y = y-1

                    cuts.append((start_y, end_y)) 
                    start_y = y
        
        for i, (start_y, end_y) in enumerate(cuts):
            if i == 0:
                cut_img = img.crop((0, 0, img.width, end_y))
            else:
                cut_img = img.crop((0, cuts[i-1][1], img.width, end_y))
            
            cut_img.save(os.path.join(output_folder, f'{i+1}.png'))

    def delete_all_folders_in(self, directory : str) -> None:
        folder_list = os.listdir(directory)

        for folder in folder_list:
            shutil.rmtree(os.path.join(directory, folder))

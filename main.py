import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
import faiss
import timm

import argparse
import asyncio
import json
import logging
import os
import platform

from aiohttp import ClientSession, web, FormData
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import ssl
import urllib

from scipy import ndimage
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

use_cuda = torch.cuda.is_available()

# 특성 벡터 차원 조절
vector_size=512
# 몇 개의 객체 탐지할지 조절 - 화면 중앙에서 가까운 순으로
target_size=3


# MongoDB에 연결
username = urllib.parse.quote_plus("jylee")
password = urllib.parse.quote_plus("memorise@106@jyjy@account")
host = "j9b106.p.ssafy.io"
port = "27017"
database = "memorise"
authDB = "admin"

connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}?authSource={authDB}"
client = MongoClient(connection_string)
db = client['memorise']
collection = db[f'yolo8_{vector_size}']

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = timm.create_model("tf_efficientnet_b2.ns_jft_in1k", pretrained=True)
        self.fc = nn.Linear(1408, vector_size)  # EfficientNetB2의 출력 차원을 1408로 수정

    def forward_one(self, x):
        x = self.base_network.forward_features(x)  # EfficientNet에서는 forward_features 메서드 사용
    
        # Global Average Pooling 적용하여 1x1x1408 형태로 변경
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    
        x = x.view(x.size(0), -1)  # 텐서를 평탄화
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


# 여러 유사도 및 거리 메트릭을 위한 유틸리티 함수 정의
def cosine_similarity(features1, features2):
    return nn.functional.cosine_similarity(features1, features2, dim=1).mean().item()

def euclidean_distance(features1, features2):
    return torch.norm(features1 - features2, dim=1).mean().item()

# 다른 거리 메트릭 시도 (예: L1 거리)
def l1_distance(features1, features2):
    return torch.norm(features1 - features2, p=1, dim=1).mean().item()

# 다른 유사도 메트릭 시도 (예: 피어슨 상관계수)
def pearson_correlation(features1, features2):
    mean1 = torch.mean(features1, dim=1)
    mean2 = torch.mean(features2, dim=1)
    cov = torch.mean((features1 - mean1) * (features2 - mean2), dim=1)
    std1 = torch.std(features1, dim=1)
    std2 = torch.std(features2, dim=1)
    return torch.mean(cov / (std1 * std2)).item()


# 유사도 및 거리 계산 함수 확장
# get_similarity 함수를 수정하여 다른 메트릭을 사용할 수 있도록 설정
def get_similarity(features1, features2, metric):
    if metric == 'cosine':
        return cosine_similarity(features1, features2)
    elif metric == 'euclidean':
        return -euclidean_distance(features1, features2)
    elif metric == 'l1':
        return -l1_distance(features1, features2)
    elif metric == 'pearson':
        return pearson_correlation(features1, features2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# YOLO8 모델 초기화
CONFIDENCE_THRESHOLD = 0.6
model = YOLO('yolov8n-seg.pt')
# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()

if use_cuda:
    model_siamese = model_siamese.cuda()

# 2. YOLO 모델을 GPU로 이동 (만약 YOLO가 GPU를 지원한다면)
if hasattr(model, 'cuda') and use_cuda:
    model = model.cuda()

# 체크포인트 파일이 존재하면 가중치를 로드합니다.
if os.path.exists(f'siamese_yolo8_{vector_size}_weights.pth'):
    model_siamese.load_state_dict(torch.load(f'siamese_yolo8_{vector_size}_weights.pth'))

model_siamese.eval()  # 평가 모드로 설정

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
relay = MediaRelay()

def process_object_region(obj_mask, orig_img):
    # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
    obj_mask_resized = cv2.resize(obj_mask, (orig_img.shape[1], orig_img.shape[0]))
    # 원시 마스크에서 1인 부분 추출
    object_region = orig_img * obj_mask_resized[:, :, np.newaxis]
    # 전처리 및 특성 추출
    object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)
    # 이미지를 uint8로 변환
    object_region_uint8 = (object_region_rgb * 255).astype(np.uint8)
    object_region_transformed = transform(object_region_uint8).unsqueeze(0)
    if use_cuda:  # 텐서를 GPU로 이동
        object_region_transformed = object_region_transformed.cuda()
    return object_region_transformed

def send_data(channel, data):
    """
    Send data to the client through the WebRTC data channel.
    Args:
        channel: The WebRTC data channel.
        data: The data to send (typically a dictionary).
    """
    channel.send(json.dumps(data))

class VideoTransformTrack2(VideoStreamTrack):
    async def send_post_request(self, url, headers, data):
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                return await response.text()

    async def send_file_post_request(self, url, file):
        data = FormData()
        data.add_field('file', file, filename='frame.jpeg', content_type='image/jpeg')
        async with ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.json()  # 응답이 JSON 형식이라고 가정


    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.frame_counter = 0  # 프레임 카운터 초기화
        self.first_frame = None  # 첫 번째 프레임을 저장하기 위한 변수를 초기화합니다.
        self.first_object = None

        # FAISS 인덱스 초기화 및 MongoDB 데이터 로드
        self.dimension = vector_size
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_db_ids = []  # DB ID 및 벡터 인덱스를 저장할 리스트
        self.vector_list = []  # 벡터를 임시 저장할 리스트
        self.image_list = []  # 벡터를 임시 저장할 리스트
        
    async def recv(self):
        frame = await self.track.recv()
        self.frame_counter += 1
        print(f"Processing frame {self.frame_counter}")

        if self.frame_counter == 1:
            self.first_frame = frame.to_ndarray(format="bgr24")  # 첫 번째 프레임을 저장합니다.

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (352, 640))
        results = model.predict(img)

        if results:
            center_x, center_y = 192, 378
            closest_obj = None
            closest_distance = float('inf')

            for i, r in list(enumerate(results)):
                if r is None or r.masks is None:
                    continue

                boxes_data = r.boxes.data.tolist()
                masks_data = r.masks.data
                orig_img = r.orig_img

                obj_idx = 0
                obj_mask = []

                for i, j in enumerate(masks_data):
                    x1, y1, x2, y2, _, _ = boxes_data[i]
                    obj_center_x = (x1 + x2) / 2
                    obj_center_y = (y1 + y2) / 2
                    distance = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)

                    if distance < closest_distance:
                        closest_distance = distance
                        obj_idx = i
                        obj_mask = j

                obj_mask = obj_mask.cpu().numpy()
                # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
                obj_mask = cv2.resize(obj_mask, (orig_img.shape[1], orig_img.shape[0]))

                # 원시 마스크에서 1인 부분 추출
                object_region = orig_img * obj_mask[:, :, np.newaxis]

                # 전처리 및 특성 추출
                object_region_rgb = cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB)

                # 이미지를 uint8로 변환
                object_region = (object_region_rgb * 255).astype(np.uint8)

                object_region = transform(object_region).unsqueeze(0).cuda()
                features_roi = model_siamese.forward_one(object_region)

                features_roi_np = features_roi.cpu().detach().numpy()
                features_vector = features_roi_np.tolist()[0]  # 중첩 리스트 제거

                # 첫 번째 프레임인 경우 벡터 리스트에 추가
                if self.frame_counter == 1:
                    self.vector_list.append(features_roi_np.tolist()[0])
                    # object_region_rgb를 image_list에 추가합니다.
                    self.image_list.append(object_region_rgb)  
                    self.first_object=object_region_rgb
                else:
                    # 가장 유사한 벡터와의 유사도 계산
                    similarities = [get_similarity(torch.tensor(vec).cuda().unsqueeze(0), features_roi, metric='cosine') 
                                    for vec in self.vector_list]
                    
                    # 유사도가 0.70 이상 0.97 이하인 경우 벡터 리스트에 추가
                    if 0.70 <= max(similarities) <= 0.93:
                        self.vector_list.append(features_roi_np.tolist()[0])
                        # object_region_rgb를 image_list에 추가합니다.
                        self.image_list.append(object_region_rgb)  

                # 벡터 리스트에 100개의 벡터가 저장되면 MongoDB에 insert
                if len(self.vector_list) == 100:
                    # 여기서 image_list에 있는 각 이미지를 회전시키고, 벡터를 추출하여 vector_list에 추가합니다.
                    for image in self.image_list:
                        for angle in [45, 90, 135, 180, 225, 270]:
                            rotated_image = ndimage.rotate(image, angle, reshape=False)
                            rotated_image_uint8 = (rotated_image * 255).astype(np.uint8)
                            rotated_image_transformed = transform(rotated_image_uint8).unsqueeze(0)
                            if use_cuda:
                                rotated_image_transformed = rotated_image_transformed.cuda()
                            rotated_features_roi = model_siamese.forward_one(rotated_image_transformed)
                            rotated_features_roi_np = rotated_features_roi.cpu().detach().numpy()
                            rotated_features_vector = rotated_features_roi_np.tolist()[0]  # 중첩 리스트 제거
                            self.vector_list.append(rotated_features_vector)  # 회전된 이미지의 벡터를 vector_list에 추가합니다.

                    db_entry = {
                        "vector": self.vector_list
                    }
                    result = collection.insert_one(db_entry)
                    self.faiss_db_ids.extend([(result.inserted_id, idx) for idx in range(100)])
                    self.faiss_index.add(np.array(self.vector_list))
                    

                    # 첫 번째 프레임을 바이트 버퍼로 변환
                    _, encoded_image = cv2.imencode('.jpeg', self.first_object)
                    byte_buffer = encoded_image.tobytes()

                    # 지정된 엔드포인트로 파일 전송
                    file_upload_response = await self.send_file_post_request(
                        "http://j9b106.p.ssafy.io:8000/items/upload", byte_buffer)

                    # 파일 업로드 응답에서 'savedFileName' 사용
                    saved_file_name = file_upload_response.get('savedFileName')

                    # 'savedFileName'을 이용하여 이미지 URL 구성
                    image_url = f"https://b106-memorise.s3.ap-northeast-2.amazonaws.com/{saved_file_name}"

                    # 이미지 URL을 사용하여 JSON POST 요청 전송
                    json_post_url = "http://j9b106.p.ssafy.io:8000/items"
                    json_post_headers = {"Content-Type": "application/json"}
                    json_post_data = {
                        "itemName": str(result.inserted_id),
                        "itemImage": image_url  # 여기에 구성된 이미지 URL 사용
                    }
                    json_post_response_text = await self.send_post_request(
                        json_post_url, json_post_headers, json_post_data)
                    
                    print(json_post_response_text)

                    data_to_send = {
                        "count": len(self.vector_list)
                    }
                    send_data(data_channel, data_to_send)

                    data_to_send = {
                        "newId": str(result.inserted_id)
                    }
                    send_data(data_channel, data_to_send)

                    # self.vector_list = []
                
                else:
                    data_to_send = {
                        "count": len(self.vector_list)
                    }
                    send_data(data_channel, data_to_send)

        return frame

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.frame_counter = 0  # <-- 프레임 카운터 초기화

        # FAISS 인덱스 초기화 및 MongoDB 데이터 로드    
        self.dimension = vector_size
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.objects_in_db = list(collection.find({}))

        # 모든 벡터를 한 리스트에 추가하고, 각 벡터에 대한 DB ID와 벡터의 인덱스를 함께 저장
        self.vectors = [vec for obj in self.objects_in_db for vec in obj['vector']]
        self.faiss_db_ids = [(obj['_id'], idx) for obj in self.objects_in_db for idx, _ in enumerate(obj['vector'])]

        # FAISS 인덱스에 벡터 추가
        if self.vectors:
            self.faiss_index.add(np.array(self.vectors))

    async def recv(self):
        frame = await self.track.recv()

        # 프레임 카운터 업데이트 및 출력
        self.frame_counter += 1
        print(f"Processing frame {self.frame_counter}")  # <-- 현재 프레임 번호 출력

        # Convert AIORTC frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (352, 640))

        results = model.predict(img)

        if self.frame_counter%5==0:
            # 화면 중앙 좌표 계산
            # center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            center_x, center_y = 192, 378
            closest_objs = []  # 화면 중앙에서 가장 가까운 객체 저장 리스트
            closest_distances = []  # 화면 중앙에서 가장 가까운 객체들과의 거리 저장 리스트

            for i, r in list(enumerate(results)):
                if r is None:
                    # cv2.imshow('YOLOv8 Object Detection', img)
                    continue
                if r.masks is None:
                    # cv2.imshow('YOLOv8 Object Detection', img)
                    continue
                
                boxes_data = r.boxes.data.tolist()
                masks_data = r.masks.data

                for obj_idx, obj_mask in enumerate(masks_data):
                    obj_mask = obj_mask.cpu().numpy()

                    # 객체 중앙 좌표 계산
                    x1, y1, x2, y2, _, _ = boxes_data[obj_idx]
                    obj_center_x = (x1 + x2) / 2
                    obj_center_y = (y1 + y2) / 2

                    # 객체와 화면 중앙 사이의 거리 계산
                    distance = np.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)

                    # 현재까지 가장 가운데에 위치한 객체인지 확인
                    if len(closest_objs) < target_size:
                        closest_objs.append((obj_idx, obj_mask))
                        closest_distances.append(distance)
                    else:
                        # 현재 객체가 가장 가까운 객체보다 더 가까울 경우, 가장 가까운 객체들 중 가장 먼 객체를 대체
                        max_distance_idx = np.argmax(closest_distances)
                        if distance < closest_distances[max_distance_idx]:
                            closest_objs[max_distance_idx] = (obj_idx, obj_mask)
                            closest_distances[max_distance_idx] = distance

                objects_data = []  # 모든 객체의 데이터를 담을 리스트를 초기화합니다.
                for obj_idx, obj_mask in closest_objs:
                    # 원시 마스크의 크기를 객체 이미지의 크기에 맞게 조절
                    features_roi = model_siamese.forward_one(process_object_region(obj_mask, r.orig_img))
                    features_roi_np = features_roi.cpu().detach().numpy()

                    # 실시간 영상에서 감지된 각 객체에 대해 가장 유사한 벡터를 faiss에서 검색
                    D, I = self.faiss_index.search(features_roi_np, k=1)
                    closest_obj_id, vector_idx = self.faiss_db_ids[I[0][0]]  # 벡터의 인덱스도 함께 가져옵니다.
                    
                    # 이미 로드된 objects_in_db에서 해당 객체를 찾습니다.
                    closest_obj = next(obj for obj in self.objects_in_db if obj['_id'] == closest_obj_id)

                    # 가장 근접한 객체의 특성 벡터로 유사도 계산
                    closest_obj_vector = torch.tensor(closest_obj['vector'][vector_idx])  # 해당 인덱스의 벡터를 가져옵니다.
                    if use_cuda:  # 텐서를 GPU로 이동
                        closest_obj_vector = closest_obj_vector.cuda()
                    closest_obj_vector = closest_obj_vector.unsqueeze(0)
                    similarity = get_similarity(closest_obj_vector, features_roi, metric='cosine')  # 'euclidean' or 'cosine' or 'l1' or 'pearson'

                    # 바운딩 박스 정보 가져오기
                    x1, y1, x2, y2, _, _ = boxes_data[obj_idx]
                    
                    # 객체 중앙 좌표 구하기
                    label_x = int((x1 + x2) // 2)
                    label_y = int((y1 + y2) // 2)

                    # 유사도가 0.50 이상인 경우에만 아이디 전달
                    if similarity >= 0.80:
                        object_data = {
                            "id": str(closest_obj['_id']),  # ObjectId를 문자열로 변환
                            "label_x": label_x,
                            "label_y": label_y
                        }
                    else:
                        object_data = {
                            "id": "0",  # ObjectId를 문자열로 변환
                            "label_x": label_x,
                            "label_y": label_y
                        }

                    objects_data.append(object_data)  # 각 객체의 데이터를 리스트에 추가합니다.

                data_to_send = {
                    "objects": objects_data  # 객체 데이터의 리스트를 보냅니다.
                }
                print(data_to_send)
                send_data(data_channel, data_to_send)  # 클라이언트로 데이터를 보냅니다.
                
        return frame

async def index(request):
    content = open(os.path.join("index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join("client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    track_type = params.get("trackType", "default")  # 기본값은 "default"

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % id(pc)

    def log_info(msg, *args):
        logging.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        global data_channel  # global 변수로 설정
        data_channel = channel  # 채널 초기화


    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            if track_type == "track2":
                pc.addTrack(VideoTransformTrack2(
                    relay.subscribe(track)
                ))
            else:  # 기본값은 VideoTransformTrack 사용
                pc.addTrack(VideoTransformTrack(
                    relay.subscribe(track)
                ))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8082, help="Port for HTTP server (default: 8080)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

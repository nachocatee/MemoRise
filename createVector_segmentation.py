import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
# from rembg import remove
from pymongo import MongoClient
import timm

# MongoDB에 연결
client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['efficientnetb2']

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
        self.fc = nn.Linear(1408, 512)  # EfficientNetB2의 출력 차원을 1408로 수정

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
    
    

# 세그멘테이션 모델 로딩
model_segmentation = torch.hub.load('ultralytics/yolov5', 'yolov5x-seg', pretrained=True)
model_segmentation.eval()


# 모델 로딩
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Siamese Network 초기화 및 가중치 로드
model_siamese = SiameseNetwork().cuda()
if os.path.exists('siamese_efficientnetb2_512_weights.pth'):
    model_siamese.load_state_dict(torch.load('siamese_efficientnetb2_512_weights.pth'))

model_siamese.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = 'C:\\Users\\SSAFY\\Desktop\\image\\StarbucksTumbler1'
all_features = []

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    local_image = cv2.imread(image_path)
    local_image_rgb = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
    
    # 세그멘테이션
    input_image = transforms.ToPILImage()(local_image_rgb)
    input_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(input_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model_segmentation(input_tensor)
        
    output_predictions = output[0].argmax(0)
    mask = output_predictions.byte().cpu().numpy()

    # 마스크를 사용하여 세그멘테이션된 객체 추출
    segmented_object = local_image_rgb * mask[:, :, None]

    # Siamese Network를 사용하여 특성 추출
    tensor_obj = transform(segmented_object).unsqueeze(0).cuda()
    features_obj = model_siamese.forward_one(tensor_obj)
    numpy_array_features = features_obj.cpu().detach().numpy()
    list_features = numpy_array_features.tolist()[0]

    all_features.append(list_features)

# 데이터베이스에 저장
db_entry = {
    "id": 1,
    "name": os.path.basename(image_folder),
    "user": "이준용",
    "vector": all_features
}

db.efficientnetb2.insert_one(db_entry)


from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import numpy as np
from FILE_YOLOv8_SEG import process_image
from FILE_YOLOv8_SEG_newObject_center import process_video
import cv2


app = FastAPI()

@app.get("/")
async def test_file():
    return "FAST API TEST"

@app.post("/check/")
async def upload_file(file: UploadFile = File(...)):
    try:
        
        # 이미지를 바이트 배열로 읽어들입니다.
        image_data = np.frombuffer(file.file.read(), np.uint8)

        # 바이트 배열에서 이미지를 로드합니다.
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        # 이미지를 처리합니다.
        result = process_image(frame)  # 이 함수는 파일 경로 대신 이미지 배열을 처리합니다.
        
        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # 비디오를 바이트 배열로 읽어들입니다.
        video_data = file.file.read()
        
        # 일시적인 파일을 생성하여 비디오 데이터를 저장합니다.
        temp_filename = "temp_video.mp4"
        with open(temp_filename, "wb") as video_file:
            video_file.write(video_data)

        # 비디오를 처리합니다.
        result = process_video(temp_filename)  # 이 함수는 비디오 파일 경로를 처리합니다.
        
        # 임시 파일을 제거합니다.
        os.remove(temp_filename)

        return result
    except Exception as e:
        return {"status": "error", "detail": str(e)}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

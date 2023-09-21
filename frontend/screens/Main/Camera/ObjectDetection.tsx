import React, { useEffect, useRef } from "react";
import {
  View,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from "react-native";
import {
  Camera,
  useCameraDevices,
  CameraDevice,
} from "react-native-vision-camera";
import Colors from "../../../constants/colors";
import axios from "axios";

const ObjectDetection: React.FC = () => {
  const devices = useCameraDevices();
  const device: CameraDevice | undefined = devices.back;
  const camera = useRef<Camera | null>(null);

  useEffect(() => {
    checkPermission();
  }, []);

  const checkPermission = async () => {
    const cameraPermission = await Camera.getCameraPermissionStatus();
    if (cameraPermission === "denied") {
      return Camera.requestCameraPermission();
    }
  };

  if (!device) return <ActivityIndicator />;

  // 사진 촬영
  const takePicture = async () => {
    if (camera.current) {
      const photo = await camera.current.takePhoto(); // 현재 카메라에서 촬영
      const ImageURI = "file:///" + photo.path; // 이미지 URI 저장

      //서버에 촬영 된 이미지 전송
      uploadImageToServer(ImageURI);
    }
  };

  // 서버로 이미지 전송 코드
  const uploadImageToServer = async (imagePath: string) => {
    // 위에서 정의한 이미지 URI 파라미터로 받음
    const formData = new FormData(); // 서버로 보낼 새로운 데이터 형식 생성

    // key값이 file인 formData 생성
    formData.append("file", {
      uri: imagePath,
      name: "temp.jpg",
      type: "image/jpeg",
    });

    try {
      const response = await axios.post(
        "http://172.20.10.4:8000/check/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log(response.data);
    } catch (error) {
      console.error("Error uploading image with axios:", error);
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        photo
      />
      <TouchableOpacity
        style={{
          width: 60,
          height: 60,
          borderRadius: 30,
          backgroundColor: Colors.primary200,
          position: "absolute",
          bottom: 50,
          alignSelf: "center",
        }}
        onPress={takePicture}
      ></TouchableOpacity>
    </View>
  );
};

export default ObjectDetection;

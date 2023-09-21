import React, { useRef, useState, useEffect } from "react";
import {
  View,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import {
  Camera,
  useCameraDevices,
  CameraDevice,
} from "react-native-vision-camera";
import axios from "axios";
import Colors from "../../../constants/colors"; // 향후 버튼 색상을 위해 추가

const ObjectRegistration: React.FC = () => {
  const devices = useCameraDevices();
  const device: CameraDevice | undefined = devices.back;
  const camera = useRef<Camera | null>(null);

  useEffect(() => {
    checkPermission();
  }, []);

  const checkPermission = async () => {
    const cameraPermission = await Camera.getCameraPermissionStatus();
    if (cameraPermission === "denied") {
      await Camera.requestCameraPermission();
    }
  };

  if (!device) return <ActivityIndicator />;

  // 녹화 여부
  const [isRecording, setIsRecording] = useState<boolean>(false);

  // 비디오 업로드
  const uploadVideo = async (videoPath: string) => {
    const videoURI = "file://" + videoPath;
    const formData = new FormData();
    formData.append("file", {
      name: "video.mp4",
      type: "video/mp4",
      uri: videoURI,
    });

    try {
      const response = await axios.post(
        "http://172.20.10.4:8000/upload/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Upload response", response.data);
    } catch (error) {
      console.error("Upload error", error);
    }
  };

  // 녹화 버튼 토글
  const toggleRecording = async () => {
    if (isRecording) {
      // 녹화 중지
      const video: any = await camera.current!.stopRecording();
      console.log("Finished recording: ", video);
      setIsRecording(false);
    } else {
      // 녹화 시도
      try {
        camera.current!.startRecording({
          flash: "on",
          onRecordingFinished: async (video: any) => {
            // 녹화가 끝나면 비디오 서버로 업로드
            await uploadVideo(video.path);
          },
          onRecordingError: (error: any) => console.error(error),
        });
        setIsRecording(true);
      } catch (e) {
        console.log(e);
      }
    }
  };

  return (
    <View style={{ flex: 1 }}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        video
      />
      <TouchableOpacity
        style={[
          styles.recordButton,
          isRecording ? styles.recording : styles.notRecording,
        ]}
        onPress={toggleRecording}
      ></TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  recordButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    position: "absolute",
    bottom: 50,
    alignSelf: "center",
  },
  recording: {
    backgroundColor: "blue", // 녹화 중일 때 파란색
  },
  notRecording: {
    backgroundColor: "red", // 녹화 중이 아닐 때 빨간색
  },
});

export default ObjectRegistration;

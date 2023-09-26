import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  Alert,
  Button,
  TouchableOpacity,
  Animated,
  ActivityIndicator,
} from "react-native";
import {
  MediaStream,
  RTCPeerConnection,
  RTCSessionDescription,
  RTCView,
  mediaDevices,
} from "react-native-webrtc";
import RNFetchBlob from "rn-fetch-blob";

import LinearGradient from "react-native-linear-gradient";

import { styles } from "./MainStyle";
import AlertModal from "../../components/Modal/AlertModal";
import { SERVER_OFFER_URL } from "../../util/http";

import axios from "axios";

const CamTestScreen = () => {
  // WebRTC 로직

  // 데이터 전송을 위한 데이터 채널 참조
  const dataChannelRef = useRef<DataChannel | null>(null);

  // 미등록된 물체에 대한 알림 여부
  const [unregisteredNotification, setUnregisteredNotification] =
    useState(false);

  // 물체 등록(현재의 RTC 연결을 종료하고 새 연결(track2)을 시작)
  const objectRegister = async () => {
    try {
      // 현재의 RTC 연결을 종료
      stopRTCConnection();

      // 새로운 서버에 연결을 시작
      await startRTCConnection("track2");

      // 미등록 물체 모달 닫기
      setUnregisteredNotification(false);
      // 물체 위 표시 삭제
      setCoordinates(null);
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  // 미등록 물체 모달 닫기
  const cancelObjectRegister = () => {
    setUnregisteredNotification(false);
  };

  // 인식 된 객체 위치 저장
  const [coordinates, setCoordinates] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);

  // RTC 서버와의 연결 상태
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // 로컬의 미디어 스트림(앱의 카메라)
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);

  // RTCPeerConnection 객체 참조
  const pc = useRef<RTCPeerConnection | null>(null);

  // SDP를 교환하는 함수
  // (비디오의 해상도, 오디오 전송 또는 수신 여부)
  const negotiate = async (trackType: string): Promise<void> => {
    try {
      // 송신 전용(sendonly)의 비디오 트랜시버 추가
      pc.current?.addTransceiver("video", { direction: "sendonly" }); //await 삭제함
      const offerOptions = {
        offerToReceiveVideo: true, // 비디오 송신
      };

      // 오퍼 생성
      const offer = await pc.current?.createOffer(offerOptions);
      if (!offer) throw new Error("Unable to create offer");

      // 로컬 SDP 설정
      await pc.current?.setLocalDescription(offer);

      // 생성된 Offer를 서버에 전송
      const response = await RNFetchBlob.config({
        //Android에서 SSL/TLS의 "Trusty" 시스템을 사용하여 보안 연결을 활성화(인증서 대체)
        trusty: true,
      }).fetch(
        "POST",
        SERVER_OFFER_URL,
        {
          "Content-Type": "application/json",
        },
        JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          trackType: trackType,
        })
      );

      // 서버로부터의 응답을 파싱하여 Answer SDP를 가져옴.
      const answer = await response.json();
      if (pc.current) {
        // 원격 SDP 설정
        await pc.current.setRemoteDescription(
          new RTCSessionDescription(answer)
        );
      }
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  // 카메라 초기화
  const initializeCamera = async (): Promise<void> => {
    try {
      const stream = await mediaDevices.getUserMedia({
        audio: true,
        video: {
          width: 3840,
          height: 2160,
          frameRate: 60,
          facingMode: "environment",
        },
      });
      setLocalStream(stream);
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  // WebRTC 연결 시작(물체 조회, 학습)
  const startRTCConnection = async (
    trackType: string = "default"
  ): Promise<void> => {
    if (!localStream) {
      Alert.alert("Error", "Please initialize the camera first.");
      return;
    }

    const configuration = {
      sdpSemantics: "unified-plan",
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      encodings: [
        {
          rid: "h",
          maxBitrate: 900000,
          minBitrate: 300000,
          maxFramerate: 30,
        },
      ],
    };

    pc.current = new RTCPeerConnection(configuration);

    // 데이터 채널 생성 및 메시지 수신 리스너 설정
    const dataChannel = pc.current.createDataChannel("data");
    dataChannelRef.current = dataChannel;
    dataChannel.onmessage = (event: any) => {
      // 서버에서 받은 데이터 처리

      const receivedData = JSON.parse(event.data);
      if (trackType != "track2") {
        const label = `Id: ${receivedData.id}, X: ${receivedData.label_x}, Y: ${receivedData.label_y}`;
        console.log(label);
        setCoordinates({
          id: receivedData.id,
          x: receivedData.label_x,
          y: receivedData.label_y,
        });
      } else {
        console.log(receivedData);
      }
    };

    // 미디어 트랙 추가
    localStream.getTracks().forEach((track) => {
      pc.current?.addTrack(track, localStream);
    });

    await negotiate(trackType); // trackType을 negotiate 함수로 전달

    setIsConnected(true);
  };

  // WebRTC 연결 종료
  const stopRTCConnection = (): void => {
    setIsConnected(false);

    if (localStream && pc.current) {
      localStream.getTracks().forEach((track) => {
        const sender = pc.current?.getSenders().find((s) => s.track === track);
        if (sender) {
          pc.current?.removeTrack(sender);
        }
      });
    }

    pc.current?.close();
    setCoordinates(null);
  };

  useEffect(() => {
    initializeCamera();
    // setUnregisteredNotification(true); // 작업을 위해 true 해놓음 추후 완전 삭제
    return () => {
      stopRTCConnection();
    };
  }, []);

  // 물체 학습 로직

  const [isLoading, setIsLoading] = useState(false);
  const rotateValue = useRef(new Animated.Value(0)).current;

  const startLoadingAnimation = () => {
    setIsLoading(true);
    Animated.loop(
      Animated.timing(rotateValue, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      })
    ).start();
  };

  const stopLoadingAnimation = () => {
    setIsLoading(false);
    rotateValue.setValue(0);
  };

  const tempRegister = () => {
    console.log("이펙트 생성");
    // 미등록 물체 모달 닫기
    setUnregisteredNotification(false);
    startLoadingAnimation();
    setTimeout(() => {
      stopLoadingAnimation();
    }, 1500);
  };
  return (
    <View style={{ flex: 1 }}>
      <View style={styles.rootContainer}>
        {/* RTC 카메라 화면 */}
        {localStream && (
          <RTCView
            style={styles.video}
            streamURL={localStream.toURL()}
            objectFit="cover"
          />
        )}

        {/* 인식된 객체에 표시할 오브젝트 */}
        {coordinates && (
          <TouchableOpacity
            style={[
              styles.ObjCircle,
              {
                left: coordinates.x,
                top: coordinates.y,
              },
            ]}
            activeOpacity={0.7} // 눌렀을 때 투명도 조절
            onPress={() => {
              if (coordinates.id !== "0") {
                // 메모 개수

                Alert.alert("Notification", "메모 개수 표시하기");
              } else {
                // 미등록 물체 알림 표시
                setUnregisteredNotification(true);
              }
            }}
          >
            <LinearGradient
              colors={["#339af0", "blue"]}
              style={styles.ObjCircleLinear}
            >
              {coordinates.id === "0" ? (
                // 등록 되지 않은 물채 표시할 텍스트
                <Text style={styles.ObjCircleText}>+</Text>
              ) : (
                // 메모 개수 표시
                <Text style={styles.ObjCircleText}>2</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>
        )}
        <AlertModal
          modalVisible={unregisteredNotification}
          closeModal={cancelObjectRegister}
          onConfirm={objectRegister} //=> 나중에 주석 해제해야함. 진짜 학습 실행
          // onConfirm={tempRegister} // 학습 실행할 시 생길 이펙트들 실행
          contentText={`미등록된 물체입니다.\n등록해주세요!`}
          btnText="확인"
        />
        <View style={styles.rtcButton}>
          <Button
            title="Start"
            onPress={() => startRTCConnection("track1")}
            disabled={isConnected}
          />
          <Button
            title="Stop"
            onPress={stopRTCConnection}
            disabled={!isConnected}
          />
        </View>
      </View>
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4C6AFF" />
          <Text style={styles.loadingText}>물체 학습을 준비 중입니다..</Text>
        </View>
      )}
    </View>
  );
};

export default CamTestScreen;

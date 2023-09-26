import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  Modal,
  Pressable,
  Image,
  ScrollView,
  Dimensions,
  Alert,
  Button,
  TouchableOpacity,
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
import { useIsFocused } from "@react-navigation/native";
import { launchImageLibrary } from "react-native-image-picker";
import axios from "axios";
import { useSelector } from "react-redux";

import MainHeader from "../../components/Header/MainHeader";
import { styles } from "./MainStyle";
import { TextInput } from "react-native-gesture-handler";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import Colors from "../../constants/colors";
import MemoBtnModal from "../../components/Modal/Memo/MemoBtnModal";
import AlertModal from "../../components/Modal/AlertModal";
import { BACKEND_URL, S3_URL, SERVER_OFFER_URL } from "../../util/http";
import MemoList from "../../components/Modal/Memo/MemoList";
import { RootState } from "../../store/store";
import MemoDetail from "../../components/Modal/Memo/MemoDetail";
import { MemoDetailProps } from "../../components/Modal/Memo/MemoDetail";

const screenHeight = Dimensions.get("window").height;

type DataChannel = {
  current: {
    _bufferedAmount: number;
    _id: number | null;
    _label: string;
    _maxPacketLifeTime?: number;
    _maxRetransmits?: number;
    _negotiated: boolean;
    _ordered: boolean;
    _peerConnectionId: number;
    _protocol: string | null;
    _reactTag: string;
    _readyState: string;
    binaryType: string;
    bufferedAmountLowThreshold: number;
  };
};

// 태그된 회원 타입
type Member = {
  [name: string]: string;
};

// 첨부 이미지 크기
const MAX_WIDTH = calculateDynamicWidth(286);
const MAX_HEIGHT = screenHeight / 2;

// 오늘 날짜 가져오기
function getFormattedDate(): string {
  const date = new Date();
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0"); // 월이 0부터 시작하므로 1을 더해줌
  const day = String(date.getDate()).padStart(2, "0");

  return `${year}. ${month}. ${day}.`;
}

const currentDate = getFormattedDate();

const MainScreen = () => {
  const isFocused = useIsFocused();
  // const token = useSelector((state: RootState) => state.userInfo.)
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 사진 첨부
  const [uploadedPic, setUploadedPic] = useState("");

  const selectImageHanlder = () => {
    launchImageLibrary(
      {
        mediaType: "photo",
        includeBase64: true,
      },
      (response: any) => {
        if (response.didCancel) {
          return;
        } else if (response.errorCode) {
          console.log("Image Error : " + response.errorCode);
        }

        // 백엔드 연동을 위한 Form Data
        const formData = new FormData();
        formData.append("file", {
          uri: response.assets[0].uri,
          type: response.assets[0].type,
          name: response.assets[0].fileName,
        });

        // S3에 사진 업로드
        axios
          .post(BACKEND_URL + "/memos/upload", formData, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          })
          .then((response: any) => {
            console.log(response);
            // 요청 성공 시, 리덕스 및 상태관리 (사용자 이미지 S3링크로 저장)
            const tempS3URL = S3_URL + response.data.savedFileName;
            console.log(tempS3URL);
            setUploadedPic(tempS3URL);
          })
          .catch((error) => {
            console.log(error);
          });
      }
    );
  };

  // 사진 상세
  const [isFullImageVisible, setFullImageVisible] = useState(false);
  const [imageWidth, setImageWidth] = useState(MAX_WIDTH);
  const [imageHeight, setImageHeight] = useState(MAX_HEIGHT);

  useEffect(() => {
    if (uploadedPic) {
      // 원본 이미지 크기
      Image.getSize(uploadedPic, (width, height) => {
        // 원본 이미지 비율 계산
        const aspectRatio = width / height;

        // 비율 유지하면서 크기 조절
        if (width >= height) {
          setImageWidth(MAX_WIDTH);
          setImageHeight(MAX_WIDTH / aspectRatio);
        } else {
          setImageHeight(MAX_HEIGHT);
          setImageWidth(MAX_HEIGHT * aspectRatio);
        }
      });
    }
  }, [uploadedPic]);

  const openFullImage = () => {
    setFullImageVisible(true);
  };

  const closeFullImage = () => {
    setFullImageVisible(false);
  };

  // 사진 삭제
  const deleteUploadedPic = () => {
    setUploadedPic("");
    setFullImageVisible(false);
  };

  // 태그된 회원 리스트
  // 더미 데이터
  const [taggedMember, setTaggedMember] = useState<Member[]>([]);

  // 검색 결과에서 태그할 유저 터치 시 실행
  const addTaggedMember = () => {
    setTaggedMember((prevData) => [
      ...prevData,
      { 권소정: "flfk33@naver.com" },
    ]);
    setSearchResultVisible(false);
    setTagSearchText("");
  };

  // 태그 검색 기능
  const [tagSearchText, setTagSearchText] = useState("");
  const [isSearchResultVisible, setSearchResultVisible] = useState(false);

  const tagSearchHandler = () => {
    setSearchResultVisible(true);
  };

  const closeTagSearch = () => {
    setSearchResultVisible(false);
    setTagSearchText("");
  };

  // 메모 조회 상태관리
  // true -> false로 변경할 것!!! <-- 변경했다면? 주석지워~
  const [memoListVisible, setMemoListVisible] = useState(false);

  // 객체에 따른 메모 조회
  const checkMemoListHandler = () => {
    setMemoListVisible(true);
  };

  // 메모모달 종료 후, 메모 작성창 띄우는 함수
  // 나중에 객체 탐지해서 메모 개수 나오면 함수 적용
  const checkMemoHandler = () => {
    setMemoListVisible(false);
    setMemoCreateModalVisible(!isMemoCreateModalVisible);
  };

  // 메모 리스트 주변 클릭 시, 모달 종료
  const closeMemoList = () => {
    setMemoListVisible(false);
  };

  // 메모 상세 모달 상태관리
  const [isMemoDetailVisible, setIsMemoDetailVisible] = useState(false);
  const [selectMemoSeq, setSelectMemoSeq] = useState<number | null>(null);

  // 메모 상세 조회 변경 함수
  const setMemoDetailModal = (memoSeq: number) => {
    setMemoListVisible(false);
    setIsMemoDetailVisible(true);
    setSelectMemoSeq(memoSeq);
  };

  // 메모 상세 주변 클릭 시, 모달 종료
  const closeMemoDetail = () => {
    setIsMemoDetailVisible(false);
  };

  // 메모 수정을 위한 상태관리
  const [isUpdateMemoTrue, setIsUpdateMemoTrue] = useState(false);

  // 메모 삭제를 위한 메모 상세 모달 상태관리 함수
  const setMemoDeleteHandler = () => {
    setIsMemoDetailVisible(false);
  };

  // 메모 수정을 위한 데이터 상태관리
  const [checkMemoDetailData, setCheckMemoDetailData] = useState<
    MemoDetailProps[]
  >([]);

  // 메모 수정
  const setMemoUpdateHandler = (data: MemoDetailProps[]) => {
    setIsMemoDetailVisible(false);
    setMemoCreateModalVisible(true);
    setIsUpdateMemoTrue(true);
    setOpenState("OPEN"); // 사용자 메모 정보에 따라 변경 예정.
    setCheckMemoDetailData(data);
  };

  const memoInputHandler = (enteredText: string) => {
    setEnteredMemo(enteredText);
  };

  // 메모 작성 및 업데이트 완료 함수
  const memoConfirmHandler = () => {
    setMemoCreateModalVisible(false);
    setIsUpdateMemoTrue(false);
    setEnteredMemo("");
    setOpenState("OPEN");
    MemoCreate();
  };

  // 메모 작성 내용
  const [enteredMemo, setEnteredMemo] = useState("");

  // 공개 범위 설정
  // OPEN: 전체공개, RESTRICT: 일부공개, CLOSED: 비공개
  const [openState, setOpenState] = useState("OPEN");
  const [isToggleOpen, setToggleOpen] = useState(false);

  const selectOpenState = () => {
    setToggleOpen(!isToggleOpen);
  };

  const chooseOpenState = (state: string) => {
    setOpenState(state);
    setToggleOpen(false);
  };

  // 메모 생성 axios
  const MemoCreate = () => {
    if (!enteredMemo) {
      Alert.alert("내용을 입력해 주세요!"); // 나중에 수정예정
    } else if (!coordinates) {
      Alert.alert("객체가 제대로 등록되지 않았습니다.");
    } else {
      axios({
        method: "POST",
        url: BACKEND_URL + `/memos`, // 물체 ID 임시로 1로 설정
        // headers: {
        //   "Content-Type": "application/json",
        //   Authorization: "Bearer " + token,
        // },
        data: {
          content: enteredMemo,
          accessType: openState,
          userId: userId,
          newFile: uploadedPic,
          itemName: coordinates.id,
        },
      })
        .then((res) => {
          if (res.request.status === 200) {
            console.log("메모 생성 성공");
          }
        })
        .catch((err) => {
          console.log(err);
        });
    }
  };

  // 메모 작성 모달
  const [isMemoCreateModalVisible, setMemoCreateModalVisible] = useState(false);

  const closeMemoCreateModal = () => {
    setMemoCreateModalVisible(false);
    setIsUpdateMemoTrue(false);
    setOpenState("OPEN");
    setToggleOpen(false);
  };

  const openMemoCreateModal = () => {
    setMemoBtnModalVisible(false);
    setMemoCreateModalVisible(true);
  };

  // 메모 작성 중 취소 확인 모달
  const [isMemoCancelModalVisible, setMemoCancelModalVisible] = useState(false);

  const openMemoCancelModal = () => {
    setMemoCancelModalVisible(true);
  };

  // 취소 버튼 눌렀을 때
  const closeMemoCancelModal = () => {
    setMemoCancelModalVisible(false);
  };

  // 확인 버튼 눌렀을 때
  const memoCancelConfirm = () => {
    setEnteredMemo("");
    setOpenState("OPEN");
    setToggleOpen(false);
    setMemoCancelModalVisible(false);
    setMemoCreateModalVisible(false);
    setIsUpdateMemoTrue(false);
    setTagSearchText("");
    setTaggedMember([]);
  };

  // 알림 모달
  const [isNotificationModalVisible, setNotificationModalVisible] =
    useState(false);

  const closeNotificationModal = () => {
    setNotificationModalVisible(false);
  };

  // 추가 버튼 모달
  const [isMemoBtnModalVisible, setMemoBtnModalVisible] = useState(false);

  const closeMemoBtnModal = () => {
    setMemoBtnModalVisible(false);
  };

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
          frameRate: 15,
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
    dataChannelRef.current = { current: dataChannel };
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

  return (
    <View style={{ flex: 1 }}>
      {!isNotificationModalVisible && (
        <View style={styles.headerContainer}>
          <MainHeader
            openModal={() => {
              setNotificationModalVisible(true);
            }}
          />
        </View>
      )}
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
                checkMemoListHandler();
                // Alert.alert("Notification", "메모 개수 표시하기");
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
        <Pressable
          style={styles.btnContainer}
          onPress={() => setMemoBtnModalVisible(true)}
        >
          <Image
            source={require("../../assets/image/mainbtn.png")}
            style={styles.addBtn}
          />
        </Pressable>
      </View>

      {/* 메모 조회 */}
      {memoListVisible && (
        <>
          <Pressable style={styles.memoClose} onPress={closeMemoList} />
          <MemoList
            onMemoWritePress={checkMemoHandler}
            onMemoDetailPress={setMemoDetailModal}
            // id={coordinates?.id}
          />
        </>
      )}

      {/* 메모 상세 조회 */}
      {isMemoDetailVisible && (
        <>
          <Pressable style={styles.memoClose} onPress={closeMemoDetail} />
          <MemoDetail
            memoSeq={selectMemoSeq}
            onMemoUpdatePress={setMemoUpdateHandler}
            onMemoDeletePress={setMemoDeleteHandler}
          />
        </>
      )}

      {/* 알림 모달 */}
      {isNotificationModalVisible && (
        <View style={styles.background}>
          <Modal
            transparent={true}
            animationType="slide"
            visible={isNotificationModalVisible}
            onRequestClose={closeNotificationModal}
          >
            {/* 헤더 */}
            <View style={styles.header}>
              <Pressable>
                <Image
                  source={require("../../assets/image/logo/logowhite.png")}
                  style={styles.logo}
                />
              </Pressable>
              <Pressable
                style={styles.cancelContainer}
                onPress={closeNotificationModal}
              >
                <Image
                  source={require("../../assets/icons/cancelwhite.png")}
                  style={styles.cancel}
                />
              </Pressable>
            </View>

            <View style={styles.modalEmptyContainer}>
              <Text style={styles.modalEmpty}>알림 없음</Text>
            </View>
          </Modal>
        </View>
      )}

      {/* 메모 버튼 모달 */}
      {isMemoBtnModalVisible && (
        <MemoBtnModal
          openMemoCreateModal={openMemoCreateModal}
          closeModal={closeMemoBtnModal}
          visible={isMemoBtnModalVisible}
        />
      )}

      {/* 메모 작성 모달 */}
      {isMemoCreateModalVisible && (
        <View style={[styles.background, { zIndex: 2 }]}>
          <Modal
            transparent={true}
            animationType="fade"
            visible={isMemoCreateModalVisible}
            onRequestClose={closeMemoCreateModal}
          >
            {/* 첨부 사진 상세 조회 */}
            {isFullImageVisible && (
              <>
                <Pressable
                  style={[
                    styles.uploadedImgBg,
                    { backgroundColor: "rgba(0, 0, 0, 0.5)", zIndex: 2 },
                  ]}
                  onPress={closeFullImage}
                />
                {/* 첨부 사진 삭제 버튼 */}
                <Pressable
                  onPress={deleteUploadedPic}
                  style={[
                    styles.binContainer,
                    {
                      transform: [
                        {
                          translateY: -(
                            imageHeight / 2 +
                            calculateDynamicWidth(26)
                          ),
                        },
                        {
                          translateX:
                            imageWidth / 2 - calculateDynamicWidth(20),
                        },
                      ],
                    },
                  ]}
                >
                  <Image
                    source={require("../../assets/icons/bin.png")}
                    style={styles.bin}
                  />
                </Pressable>
                {/* 첨부 사진 */}
                <Image
                  source={{ uri: uploadedPic }}
                  style={[
                    styles.uploadedFullImg,
                    {
                      width: imageWidth,
                      height: imageHeight,
                      transform: [
                        { translateY: -imageHeight / 2 },
                        { translateX: -imageWidth / 2 },
                      ],
                    },
                  ]}
                />
              </>
            )}
            <Pressable
              style={{ flex: 1, backgroundColor: "transparent" }}
              onPress={openMemoCancelModal}
            />
            {/* 유저 태그(empty) */}
            {openState === "RESTRICT" && !isSearchResultVisible && (
              <View style={styles.tagContainer}>
                <View style={styles.tagSearchContainer}>
                  <Text style={styles.tagText}>@</Text>
                  <TextInput
                    style={styles.tagText}
                    placeholder="태그할 닉네임이나 그룹명을 입력해 주세요  "
                    placeholderTextColor="rgba(44, 44, 44, 0.5)"
                    value={tagSearchText}
                    onChangeText={setTagSearchText}
                    returnKeyType="search"
                    onSubmitEditing={tagSearchHandler}
                  />
                </View>
              </View>
            )}
            {/* 유저 태그(결과값 O) */}
            {/* 더미 데이터 */}
            {openState === "RESTRICT" && taggedMember[0] && (
              <>
                <ScrollView horizontal style={styles.tagResultBox}>
                  <View
                    style={{
                      justifyContent: "flex-start",
                      alignItems: "center",
                      flexDirection: "row",
                    }}
                  >
                    <LinearGradient
                      colors={["#DDEAFF", "#C2D8FF"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 0, y: 1 }}
                      style={styles.taggedMemberContainer}
                    >
                      <Text style={styles.tagText}>
                        @ {Object.keys(taggedMember[0])[0]}
                      </Text>
                      <Pressable>
                        <Image
                          source={require("../../assets/icons/cancel_sm.png")}
                          style={styles.cancelIcon}
                        />
                      </Pressable>
                    </LinearGradient>
                    <LinearGradient
                      colors={["#DDEAFF", "#C2D8FF"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 0, y: 1 }}
                      style={styles.taggedMemberContainer}
                    >
                      <Text style={styles.tagText}>
                        @ {Object.keys(taggedMember[0])[0]}
                      </Text>
                      <Pressable>
                        <Image
                          source={require("../../assets/icons/cancel_sm.png")}
                          style={styles.cancelIcon}
                        />
                      </Pressable>
                    </LinearGradient>
                    <LinearGradient
                      colors={["#DDEAFF", "#C2D8FF"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 0, y: 1 }}
                      style={styles.taggedMemberContainer}
                    >
                      <Text style={styles.tagText}>
                        @ {Object.keys(taggedMember[0])[0]}
                      </Text>
                      <Pressable>
                        <Image
                          source={require("../../assets/icons/cancel_sm.png")}
                          style={styles.cancelIcon}
                        />
                      </Pressable>
                    </LinearGradient>
                    <LinearGradient
                      colors={["#DDEAFF", "#C2D8FF"]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 0, y: 1 }}
                      style={styles.taggedMemberContainer}
                    >
                      <Text style={styles.tagText}>
                        @ {Object.keys(taggedMember[0])[0]}
                      </Text>
                      <Pressable>
                        <Image
                          source={require("../../assets/icons/cancel_sm.png")}
                          style={styles.cancelIcon}
                        />
                      </Pressable>
                    </LinearGradient>
                  </View>
                </ScrollView>
              </>
            )}
            {/* 검색 결과 */}
            {isSearchResultVisible && (
              <>
                <Pressable
                  style={styles.closeTagSearch}
                  onPress={closeTagSearch}
                />
                <View style={styles.tagResultContainer}>
                  <View
                    style={[
                      styles.tagSearchContainer,
                      {
                        borderBottomWidth: 1,
                        borderBottomColor: "rgba(44, 44, 44, 0.5)",
                      },
                    ]}
                  >
                    <Text style={styles.tagText}>@</Text>
                    <TextInput
                      style={styles.tagText}
                      placeholder="태그할 닉네임이나 그룹명을 입력해 주세요  "
                      placeholderTextColor="rgba(44, 44, 44, 0.5)"
                      value={tagSearchText}
                      onChangeText={setTagSearchText}
                      returnKeyType="search"
                      onSubmitEditing={tagSearchHandler}
                    />
                  </View>
                  {/* 더미데이터 */}
                  <Pressable
                    style={styles.tagResultInnerContainer}
                    onPress={addTaggedMember}
                  >
                    <Text style={styles.tagText}>
                      권소정 <Text style={styles.email}> flfk33@naver.com</Text>
                    </Text>
                  </Pressable>
                  <Pressable style={styles.tagResultInnerContainer}>
                    <Text style={styles.tagText} onPress={addTaggedMember}>
                      권소정{" "}
                      <Text style={styles.email}> bijoucastle@naver.com</Text>
                    </Text>
                  </Pressable>
                </View>
              </>
            )}
            <View style={styles.memoContainer}>
              <LinearGradient
                colors={["#FFFFFF", "#F5F5F5"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 0, y: 1 }}
                style={{ flex: 1 }}
              >
                <View style={{ padding: calculateDynamicWidth(15) }}>
                  {/* 공개 범위 설정 버튼 */}
                  <View style={styles.memoInnerContainer}>
                    <View>
                      {openState === "OPEN" && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/public.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {openState === "RESTRICT" && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/restrict.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {openState === "CLOSED" && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/closed.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {isToggleOpen && (
                        <View style={styles.toggleContainer}>
                          <Pressable onPress={() => chooseOpenState("OPEN")}>
                            <View style={styles.toggleContentContainer}>
                              <Text style={styles.toggleText}>전체공개</Text>
                              {openState === "OPEN" && (
                                <View style={styles.blueDotContainer}>
                                  <View style={styles.blueDot}></View>
                                </View>
                              )}
                            </View>
                          </Pressable>
                          <Pressable
                            onPress={() => chooseOpenState("RESTRICT")}
                          >
                            <View style={styles.toggleContentContainer}>
                              <Text style={styles.toggleText}>일부공개</Text>
                              {openState === "RESTRICT" && (
                                <View style={styles.blueDotContainer}>
                                  <View style={styles.blueDot}></View>
                                </View>
                              )}
                            </View>
                          </Pressable>
                          <Pressable onPress={() => chooseOpenState("CLOSED")}>
                            <View
                              style={[
                                styles.toggleClosedContentContainer,
                                { paddingLeft: calculateDynamicWidth(16) },
                              ]}
                            >
                              <Text style={styles.toggleText}>비공개</Text>
                              {openState === "CLOSED" && (
                                <View style={styles.blueDotContainer}>
                                  <View
                                    style={[
                                      styles.blueDot,
                                      { backgroundColor: Colors.text },
                                    ]}
                                  ></View>
                                </View>
                              )}
                            </View>
                          </Pressable>
                        </View>
                      )}

                      <Text style={styles.currentDate}>{currentDate}</Text>
                    </View>
                    <View style={styles.memoInnerBtnContainer}>
                      <Pressable onPress={selectImageHanlder}>
                        <Image
                          source={require("../../assets/icons/addpic.png")}
                          style={styles.addPic}
                        />
                      </Pressable>
                      <Pressable onPress={memoConfirmHandler}>
                        <View>
                          <Image
                            source={require("../../assets/icons/confirm.png")}
                            style={styles.confirm}
                          />
                        </View>
                      </Pressable>
                    </View>
                  </View>
                  <ScrollView>
                    {uploadedPic && (
                      <Pressable
                        style={{ alignItems: "center" }}
                        onPress={openFullImage}
                      >
                        <Image
                          source={{ uri: uploadedPic }}
                          style={styles.uploadedImg}
                        />
                      </Pressable>
                    )}
                    {isUpdateMemoTrue ? (
                      // 더미데이터
                      <Text>테스트!</Text>
                    ) : (
                      <TextInput
                        style={styles.memoContent}
                        multiline={true}
                        onChangeText={memoInputHandler}
                        value={enteredMemo}
                      />
                    )}
                  </ScrollView>
                </View>
              </LinearGradient>
            </View>
          </Modal>
        </View>
      )}

      {/* 메모 작성 중 취소 확인 모달 */}
      {isMemoCancelModalVisible && (
        <AlertModal
          modalVisible={isMemoCancelModalVisible}
          closeModal={closeMemoCancelModal}
          onConfirm={memoCancelConfirm}
          contentText="메모 작성을 취소하시겠습니까?"
          btnText="확인"
        />
      )}
    </View>
  );
};

export default MainScreen;

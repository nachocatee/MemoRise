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
import * as Progress from "react-native-progress";

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
import { GroupData } from "../MenuContents/Group/MyGroupScreen";
import TutorialModal from "../../components/Modal/TutorialModal";
import ConfirmBtn from "../../components/Button/ConfirmBtn";

const screenHeight = Dimensions.get("window").height;

export type DataChannel = {
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

// 태그 시 유저 검색 리스트
type UserData = {
  userSeq: number;
  nickname: string;
  email: string;
}[];

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
  // 메모 조회에서 생성하는지 학습에서 생성하는지 여부 확인
  const [check, setCheck] = useState(false);
  const isFocused = useIsFocused();
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 사진 첨부
  const [uploadedPic, setUploadedPic] = useState<string | null>(null);

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
            // 요청 성공 시, 리덕스 및 상태관리 (사용자 이미지 S3링크로 저장)
            const tempS3URL = S3_URL + response.data.savedFileName;
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
    setUploadedPic(null);
    setFullImageVisible(false);
  };

  // 태그된 회원 리스트
  const [taggedMember, setTaggedMember] = useState<number[]>([]);
  const [taggedMemberList, setTaggedMemberList] = useState<
    { id: number; name: string }[]
  >([]);

  // 태그된 그룹 리스트
  const [taggedGroup, setTaggedGroup] = useState<number[]>([]);
  const [taggedGroupList, setTaggedGroupList] = useState<
    { id: number; name: string }[]
  >([]);

  // 검색 결과에서 태그할 유저 터치 시 실행
  const addTaggedMember = (userSeq: number, userName: string) => {
    setTaggedMember((prevData) => {
      if (!prevData.includes(userSeq)) {
        // userSeq가 prevData에 없을 경우에만 추가
        return [...prevData, userSeq];
      }
      return prevData; // 이미 존재하면 prevData 반환
    });

    setTaggedMemberList((prevData) => {
      if (!prevData.some((user) => user.id === userSeq)) {
        return [...prevData, { id: userSeq, name: userName }];
      }
      return prevData; // 이미 존재하면 prevData 반환
    });
  };

  // 내 그룹 목록에서 태그할 그룹 터치 시 실행
  const addTaggedGroup = (teamSeq: number, teamName: string) => {
    setTaggedGroup((prevData) => {
      if (!prevData.includes(teamSeq)) {
        // teamSeq가 prevData에 없을 경우에만 추가
        return [...prevData, teamSeq];
      }
      return prevData; // 이미 존재하면 prevData 반환
    });

    setTaggedGroupList((prevData) => {
      // teamSeq가 prevData에 없을 경우에만 추가
      if (!prevData.some((group) => group.id === teamSeq)) {
        return [...prevData, { id: teamSeq, name: teamName }];
      }
      return prevData; // 이미 존재하면 prevData 반환
    });
  };

  // 그룹 태그 삭제
  const removeTaggedGroup = (groupId: number) => {
    setTaggedGroup((prevData) => {
      const newData = prevData.filter((teamSeq) => teamSeq !== groupId);
      return newData;
    });

    setTaggedGroupList((prevData) => {
      const newData = prevData.filter((group) => group.id !== groupId);
      return newData;
    });
  };

  // 유저 태그 삭제
  const removeTaggedMember = (memberId: number) => {
    setTaggedMember((prevData) => {
      const newData = prevData.filter((userSeq) => userSeq !== memberId);
      return newData;
    });

    setTaggedMemberList((prevData) => {
      const newData = prevData.filter((user) => user.id !== memberId);
      return newData;
    });
  };

  // 태그 검색 기능
  const [tagSearchText, setTagSearchText] = useState("");
  const [isSearchResultVisible, setSearchResultVisible] = useState(false);

  const closeTagSearch = () => {
    setSearchResultVisible(false);
    setTagSearchText("");
    setMyGroupVisible(false);
  };

  // 유저 검색
  const [userList, setUserList] = useState<UserData | null>(null);

  const searchUserHandler = async () => {
    if (tagSearchText) {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + "/user/list",
          params: {
            keyword: tagSearchText,
          },
        });
        setUserList(res.data);
        setSearchResultVisible(true);
      } catch (err) {
        console.log(err);
      }
    } else {
      Alert.alert("닉네임이나 이메일을 입력해 주세요!");
    }
  };

  // 메모 조회 상태관리
  // true -> false로 변경할 것!!! <-- 변경했다면? 주석지워~
  const [memoListVisible, setMemoListVisible] = useState(false);

  // 메모모달 종료 후, 메모 작성창 띄우는 함수
  // 나중에 객체 탐지해서 메모 개수 나오면 함수 적용
  const checkMemoHandler = () => {
    setMemoListVisible(false);
    setMemoCreateModalVisible(!isMemoCreateModalVisible);
  };

  // 메모 리스트 주변 클릭 시, 모달 종료
  // 여기다가 물체 조회 추가해야함
  const closeMemoList = () => {
    setMemoListVisible(false);
    // 다시 물체 표시 생성
    setIsVisible(true);
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
    setMemoListVisible(true);
  };

  // 메모 수정을 위한 상태관리 (나중에 필요없다고 판단되면 삭제할 것!)
  const [isUpdateMemoTrue, setIsUpdateMemoTrue] = useState(false);

  // 메모 삭제를 위한 메모 상세 모달 상태관리 함수
  const setMemoDeleteHandler = () => {
    setIsMemoDetailVisible(false);
  };

  // 메모 수정을 위한 데이터 상태관리 (나중에 필요없다고 판단되면 삭제할 것!)
  const [checkMemoDetailData, setCheckMemoDetailData] = useState<
    MemoDetailProps[]
  >([]);

  // 메모수정을 위한 태그된 팀 & 유저 함수
  const memoUpdateTaggedHandler = (data: MemoDetailProps[]) => {
    const userIds: number[] = [];
    const userNicknamesAndIds: { id: number; name: string }[] = [];
    const teamIds: number[] = [];
    const teamNicknamesAndIds: { id: number; name: string }[] = [];

    data[0].taggedUserList.forEach((user) => {
      userIds.push(user.userSeq);
      userNicknamesAndIds.push({ id: user.userSeq, name: user.nickname });
    });

    data[0].taggedTeamList.forEach((team) => {
      teamIds.push(team.teamSeq);
      teamNicknamesAndIds.push({ id: team.teamSeq, name: team.name });
    });

    setTaggedMember(userIds);
    setTaggedMemberList(userNicknamesAndIds);
    setTaggedGroup(teamIds);
    setTaggedGroupList(teamNicknamesAndIds);
  };

  // 메모수정에 필요한 메모ID 상태관리
  const [memoId, setMemoId] = useState<number | null>(null);

  // 메모 수정
  const setMemoUpdateHandler = (data: MemoDetailProps[]) => {
    setIsMemoDetailVisible(false);
    setMemoCreateModalVisible(true);
    setIsUpdateMemoTrue(true);
    setCheckMemoDetailData(data);
    setEnteredMemo(data[0].content);
    setOpenState(data[0].accessType);
    memoUpdateTaggedHandler(data);
    setMemoId(data[0].memoSeq);
  };

  const memoInputHandler = (enteredText: string) => {
    setEnteredMemo(enteredText);
  };

  // 메모 작성 및 업데이트 완료 함수
  const memoConfirmHandler = () => {
    if (isUpdateMemoTrue) {
      setIsUpdateMemoTrue(false);
      setIsVisible(false);
    } else {
      setIsVisible(true);
    }
    setMemoListVisible(true);
    setMemoCreateModalVisible(false);
    setEnteredMemo("");
    setOpenState("OPEN");
    MemoCreate();
    setUploadedPic("");

    if (check) {
      startRTCConnection("track1");
    }
  };

  // 메모 작성 내용
  const [enteredMemo, setEnteredMemo] = useState("");

  // 공개 범위 설정 & 태그 시 내 그룹 목록 불러오기
  // OPEN: 전체공개, RESTRICT: 일부공개, CLOSED: 비공개
  const [openState, setOpenState] = useState("OPEN");
  const [isToggleOpen, setToggleOpen] = useState(false);
  const [groupList, setGroupList] = useState<GroupData | null>(null);
  const [isMyGroupVisible, setMyGroupVisible] = useState(false);

  const selectOpenState = () => {
    setToggleOpen(!isToggleOpen);
  };

  const chooseOpenState = (state: string) => {
    setOpenState(state);
    setToggleOpen(false);
  };

  // 내 그룹 보여주기
  const openMyGroupList = () => {
    setMyGroupVisible(true);
    getGroupList();
  };

  // 태그 시 내 그룹 목록 불러오기
  const getGroupList = async () => {
    try {
      const res = await axios({
        method: "GET",
        url: BACKEND_URL + `/user/${userId}/my-teams`,
      });
      setGroupList(res.data);
    } catch (err) {
      console.log(err);
    }
  };

  // 메모 생성 axios
  const MemoCreate = async () => {
    if (!enteredMemo) {
      Alert.alert("내용을 입력해 주세요!");
    } else if (!coordinates) {
      Alert.alert("객체가 제대로 등록되지 않았습니다.");
    } else {
      if (!isUpdateMemoTrue) {
        await axios({
          method: "POST",
          url: BACKEND_URL + `/memos`,
          data: {
            content: enteredMemo,
            accessType: openState,
            userId: userId,
            newFile: uploadedPic,
            itemName: pickItem,
            taggedUserList: taggedMember,
            taggedTeamList: taggedGroup,
          },
        })
          .then((res) => {
            if (res.request.status === 200) {
              //물체 표시 생성
              setTaggedMember([]);
              setTaggedGroup([]);
              setTaggedMemberList([]);
              setTaggedGroupList([]);

              // 메모 카운트 업데이트
              getObjMemoCount();
            }
          })
          .catch((err) => {
            console.log(err);
          });
      } else {
        await axios({
          method: "PUT",
          url: BACKEND_URL + `/memos/${memoId}`,
          data: {
            content: enteredMemo,
            accessType: openState,
            userId: userId,
            newFile: uploadedPic,
            itemName: pickItem,
            taggedUserList: taggedMember,
            taggedTeamList: taggedGroup,
          },
        })
          .then((res) => {
            if (res.request.status === 200) {
              //물체 표시 생성
              setTaggedMember([]);
              setTaggedGroup([]);
              setTaggedMemberList([]);
              setTaggedGroupList([]);
              setIsUpdateMemoTrue(false);

              // 메모 카운트 업데이트
              getObjMemoCount();
            }
          })
          .catch((err) => {
            console.log(err);
          });
      }
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
    if (isUpdateMemoTrue) {
      setIsUpdateMemoTrue(false);
      setIsMemoDetailVisible(true);
    }
    setEnteredMemo("");
    setOpenState("OPEN");
    setToggleOpen(false);
    setMemoCancelModalVisible(false);
    setMemoCreateModalVisible(false);
    setTagSearchText("");
    setTaggedMember([]);
    setMyGroupVisible(false);
    setTaggedMember([]);
    setTaggedGroup([]);
    setTaggedMemberList([]);
    setTaggedGroupList([]);

    //물체 표시 생성
    setIsVisible(true);
  };

  // 알림 모달
  const [isNotificationModalVisible, setNotificationModalVisible] =
    useState(false);

  const closeNotificationModal = () => {
    setNotificationModalVisible(false);
    startRTCConnection("track2");
  };

  // 추가 버튼 모달
  const [isMemoBtnModalVisible, setMemoBtnModalVisible] = useState(false);

  const closeMemoBtnModal = () => {
    setMemoBtnModalVisible(false);
  };

  // WebRTC 로직
  // 인식 된 객체 위치 저장
  const [coordinates, setCoordinates] = useState<
    Array<{ id: string; x: number; y: number }>
  >([]);

  // 선택 된 객체 ID 값 저장
  const [pickItem, setPickItem] = useState("");

  // 메모 조회 시 객체 표시 가리기
  const [isVisible, setIsVisible] = useState(false);

  // 물체 학습 화면 표시 여부
  const [objectRegisterShow, setObjectRegisterShow] = useState(false);

  // 물체 등록 진행 상태
  const [progress, setProgress] = useState(0);

  // 물체 등록된 ID 값
  const [newId, setNewId] = useState<string | null>(null);

  // RTC 서버와의 연결 상태
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // 로컬의 미디어 스트림(앱의 카메라)
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);

  // RTCPeerConnection 객체 참조
  const pc = useRef<RTCPeerConnection | null>(null);

  // 데이터 전송을 위한 데이터 채널 참조
  const dataChannelRef = useRef<DataChannel | null>(null);

  // 미등록된 물체에 대한 알림 여부
  const [unregisteredNotification, setUnregisteredNotification] =
    useState(false);

  // 물체 등록 후 메모 생성 여부 확인
  const [confirmMemoCreate, setConfirmMemoCreate] = useState(false);

  //  memoListVisible 참조
  const memoListVisibleRef = useRef(memoListVisible);

  // memoListVisible 값이 변경될 때마다 useRef 업데이트
  useEffect(() => {
    memoListVisibleRef.current = memoListVisible;
  }, [memoListVisible]);

  // 물체 등록 후 메모 생성 취소
  const cancelConfirmMemoCreate = () => {
    startRTCConnection("track1");
    setConfirmMemoCreate(false);
  };

  // 물체 등록 후 메모 생성
  const objMemoCreate = () => {
    setConfirmMemoCreate(false);
    openMemoCreateModal();
    setCheck(true);
  };

  // 물체 등록(현재의 RTC 연결을 종료하고 새 연결(track2)을 시작)
  const objectRegister = async () => {
    try {
      // 현재의 RTC 연결을 종료
      stopRTCConnection();

      // 미등록 물체 모달 닫기
      setUnregisteredNotification(false);

      // 물체 표시 닫기
      setIsVisible(false);

      // 물체 등록 화면 생성
      setObjectRegisterShow(true);

      // 새로운 서버에 연결을 시작
      await startRTCConnection("track2");
    } catch (error) {
      Alert.alert("Error", (error as Error).message);
    }
  };

  // 미등록 물체 모달 닫기
  const cancelObjectRegister = () => {
    setUnregisteredNotification(false);
    setIsVisible(true);
  };

  type ObjMemoCountItem = {
    itemName: string;
    countMemo: number;
  };

  // 물체에 담긴 메모 개수
  const [objMemoCount, setObjMemoCount] = useState<ObjMemoCountItem[]>([]);

  const getObjMemoCount = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/memos/all/${userId}`);

      setObjMemoCount(response.data);
    } catch (error) {
      console.error("물체에 담긴 메모 개수를 가져오지 못하였습니다", error);
    }
  };

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
        audio: false,
        video: {
          width: 1280,
          height: 720,
          frameRate: 20,
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
    setIsVisible(true);
    const configuration = {
      sdpSemantics: "unified-plan",
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      encodings: [
        {
          rid: "h",
          maxBitrate: 900000,
          minBitrate: 300000,
          maxFramerate: 20,
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
      console.log(receivedData);

      if (receivedData.count && typeof receivedData.count === "number") {
        // count 값에 따라 progress 업데이트 (1-9를 0.1-0.9로 변환)
        setIsVisible(false);
        setProgress(receivedData.count / 10);
      }

      if (receivedData.newId && typeof receivedData.newId === "string") {
        // newid 값을 상태에 저장
        setPickItem(receivedData.newId);
        setNewId(receivedData.newId);
        setIsVisible(false);
        setProgress(0);
        stopRTCConnection();
        setConfirmMemoCreate(true);
      }

      if (
        trackType != "track2" &&
        receivedData.objects &&
        Array.isArray(receivedData.objects) &&
        !memoListVisibleRef.current
      ) {
        const newCoordinates = receivedData.objects.map((obj: any) => ({
          id: obj.id,
          x: obj.label_x,
          y: obj.label_y,
        }));
        setCoordinates(newCoordinates);
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
    setObjectRegisterShow(false);

    if (localStream && pc.current) {
      localStream.getTracks().forEach((track) => {
        const sender = pc.current?.getSenders().find((s) => s.track === track);
        if (sender) {
          pc.current?.removeTrack(sender);
        }
      });
    }

    pc.current?.close();
    setCoordinates([]);
    setIsVisible(false);
  };

  useEffect(() => {
    initializeCamera();
    getObjMemoCount(); // 물체에 담긴 메모 개수 가져옴
    // setUnregisteredNotification(true); // 작업을 위해 true 해놓음 추후 완전 삭제
    return () => {
      stopRTCConnection();
    };
  }, []);

  useEffect(() => {
    if (localStream) {
      startRTCConnection("track1");
    }
  }, [localStream]);

  const [checkMemoLengthZero, setCheckMemoLengthZero] = useState(false);

  const checkMemoLength = (checker: boolean) => {
    setCheckMemoLengthZero(checker);
  };

  const noLengthModalClose = () => {
    setCheckMemoLengthZero(false);
  };

  const noLengthModalOpenCreate = () => {
    setMemoCreateModalVisible(true);
    setCheckMemoLengthZero(false);
  };

  // 튜토리얼
  const [isTutorialVisible, setTutorialVisible] = useState(false);
  const [page, setPage] = useState(1);

  return (
    <View style={{ flex: 1 }}>
      {!isNotificationModalVisible && (
        <View style={styles.headerContainer}>
          <MainHeader
            openModal={() => {
              setTutorialVisible(!isTutorialVisible);
              setPage(1);
            }}
            stopRTC={() => stopRTCConnection()}
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
        {isVisible &&
          coordinates.length > 0 &&
          coordinates.map((coordinate, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.ObjCircle,
                {
                  left: coordinate.x,
                  top: coordinate.y,
                },
              ]}
              activeOpacity={0.7}
              onPress={() => {
                if (coordinate.id !== "0") {
                  setIsVisible(false);
                  setCheck(false);
                  setPickItem(coordinate.id);
                  setMemoListVisible(true);
                } else {
                  setIsVisible(false);
                  setUnregisteredNotification(true);
                }
              }}
            >
              <View style={styles.imageContainer}>
                <Image
                  source={require("../../assets/focus/memo.png")}
                  style={[styles.ObjImg, { opacity: 0.7 }]}
                />
                <View style={styles.textContainer}>
                  <Text style={styles.ObjCircleText}>
                    {coordinate.id === "0"
                      ? "+"
                      : objMemoCount.find(
                          (item) => item.itemName === coordinate.id
                        )?.countMemo || 0}
                  </Text>
                </View>
              </View>
            </TouchableOpacity>
          ))}
        {/* 물체 학습화면 로직 구현중 */}

        {objectRegisterShow && (
          <>
            <View style={styles.focusBox}>
              <Image
                source={require("../../assets/focus/focus2new.png")}
                style={styles.focusImg}
              />
            </View>
            <View style={styles.descriptionBox}>
              <Image source={require("../../assets/image/warning.png")} />
              <Text style={styles.descriptionText}>
                물체를 다각도로 촬영해주세요!
              </Text>
            </View>
            <View style={styles.progressBox}>
              <Progress.Bar
                progress={progress}
                width={300}
                height={20}
                borderRadius={10}
                borderWidth={3}
                borderColor="white"
                style={styles.progressBar}
              />
              <Text style={styles.progressText}>
                {Math.round(progress * 100)}%
              </Text>
            </View>
          </>
        )}

        <AlertModal
          modalVisible={confirmMemoCreate}
          closeModal={cancelConfirmMemoCreate}
          onConfirm={objMemoCreate}
          contentText={`물체 등록이 완료되었습니다!\n메모를 생성하시겠습니까?`}
          btnText="확인"
        />

        <AlertModal
          modalVisible={unregisteredNotification}
          closeModal={cancelObjectRegister}
          onConfirm={objectRegister}
          contentText={`미등록된 물체입니다.\n등록해주세요!`}
          btnText="확인"
        />
        {isConnected ? (
          <Pressable style={styles.btnContainer} onPress={stopRTCConnection}>
            <Image
              source={require("../../assets/image/camerabtn_stop.png")}
              style={styles.addBtn}
            />
          </Pressable>
        ) : (
          <Pressable
            style={styles.btnContainer}
            onPress={() => setNotificationModalVisible(true)}
          >
            <Image
              source={require("../../assets/image/camerabtn.png")}
              style={styles.addBtn}
            />
          </Pressable>
        )}
        {/* <Pressable
          style={styles.btnContainer}
          onPress={() => setMemoBtnModalVisible(true)}
        >
          <Image
            source={require("../../assets/image/mainbtn.png")}
            style={styles.addBtn}
          />
        </Pressable> */}
      </View>

      {/* 메모 조회 */}
      {coordinates && memoListVisible && (
        <>
          <Pressable style={styles.memoClose} onPress={closeMemoList} />
          <MemoList
            onMemoWritePress={checkMemoHandler}
            onMemoDetailPress={setMemoDetailModal}
            id={pickItem}
            memoStatus={"main"}
            memoListStatus={memoListVisible}
            onMemoZeroLength={checkMemoLength}
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

      {/* 튜토리얼 모달 */}
      {isNotificationModalVisible && (
        <TutorialModal
          modalVisible={isNotificationModalVisible}
          closeModal={closeNotificationModal}
          onlyCloseModal={() => setNotificationModalVisible(false)}
        />
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
                {uploadedPic && (
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
                )}
              </>
            )}
            <Pressable
              style={{ flex: 1, backgroundColor: "transparent" }}
              onPress={openMemoCancelModal}
            />
            {/* 유저 태그(empty) */}
            {openState === "RESTRICT" && !isSearchResultVisible && (
              <>
                {isMyGroupVisible ? (
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
                          placeholder="태그할 그룹을 선택하거나 유저를 검색해 주세요  "
                          placeholderTextColor="rgba(44, 44, 44, 0.5)"
                          value={tagSearchText}
                          onChangeText={setTagSearchText}
                          returnKeyType="search"
                          onSubmitEditing={searchUserHandler}
                        />
                      </View>
                      {/* 내 그룹 목록 */}
                      {groupList?.map((group, idx) => (
                        <Pressable
                          style={styles.tagResultInnerContainer}
                          onPress={() =>
                            addTaggedGroup(group.teamSeq, group.teamName)
                          }
                          key={idx}
                        >
                          <Text style={styles.tagText}>
                            {group.teamName}{" "}
                            <Text style={styles.email}> #{group.teamSeq}</Text>
                          </Text>
                        </Pressable>
                      ))}
                    </View>
                  </>
                ) : (
                  <View style={styles.tagContainer}>
                    <Pressable
                      onPress={openMyGroupList}
                      style={styles.tagSearchContainer}
                    >
                      <Text style={styles.tagText}>
                        @{" "}
                        <Text style={{ color: "rgba(44, 44, 44, 0.5)" }}>
                          태그할 그룹을 선택하거나 유저를 검색해 주세요
                        </Text>
                      </Text>
                    </Pressable>
                  </View>
                )}
              </>
            )}
            {/* 유저 태그(결과값 O) */}
            {openState === "RESTRICT" && (
              <>
                <ScrollView horizontal style={styles.tagResultBox}>
                  {/* 그룹 태그 */}
                  {taggedGroupList.map((group, idx) => (
                    <View
                      style={{
                        justifyContent: "flex-start",
                        alignItems: "center",
                        flexDirection: "row",
                      }}
                      key={idx}
                    >
                      <LinearGradient
                        colors={["#DDEAFF", "#C2D8FF"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 0, y: 1 }}
                        style={styles.taggedMemberContainer}
                      >
                        <Text style={styles.tagText}>@ {group.name}</Text>
                        <Pressable onPress={() => removeTaggedGroup(group.id)}>
                          <Image
                            source={require("../../assets/icons/cancel_sm.png")}
                            style={styles.cancelIcon}
                          />
                        </Pressable>
                      </LinearGradient>
                    </View>
                  ))}
                  {/* 유저 태그 */}
                  {taggedMemberList.map((member, idx) => (
                    <View
                      style={{
                        justifyContent: "flex-start",
                        alignItems: "center",
                        flexDirection: "row",
                      }}
                      key={idx}
                    >
                      <LinearGradient
                        colors={["#DDEAFF", "#C2D8FF"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 0, y: 1 }}
                        style={styles.taggedMemberContainer}
                      >
                        <Text style={styles.tagText}>@ {member.name}</Text>
                        <Pressable
                          onPress={() => removeTaggedMember(member.id)}
                        >
                          <Image
                            source={require("../../assets/icons/cancel_sm.png")}
                            style={styles.cancelIcon}
                          />
                        </Pressable>
                      </LinearGradient>
                    </View>
                  ))}
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
                      placeholder="태그할 그룹을 선택하거나 유저를 검색해 주세요  "
                      placeholderTextColor="rgba(44, 44, 44, 0.5)"
                      value={tagSearchText}
                      onChangeText={setTagSearchText}
                      returnKeyType="search"
                      onSubmitEditing={searchUserHandler}
                    />
                  </View>
                  {/* 유저 검색 결과 */}
                  {Array.isArray(userList) && userList[0] ? (
                    userList?.map((user, idx) => (
                      <Pressable
                        style={styles.tagResultInnerContainer}
                        onPress={() =>
                          addTaggedMember(user.userSeq, user.nickname)
                        }
                        key={idx}
                      >
                        <Text style={styles.tagText}>
                          {user.nickname}{" "}
                          <Text style={styles.email}> {user.email}</Text>
                        </Text>
                      </Pressable>
                    ))
                  ) : (
                    <View style={styles.tagResultInnerContainer}>
                      <Text style={styles.tagText}>검색 결과가 없습니다.</Text>
                    </View>
                  )}
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
                    <TextInput
                      style={styles.memoContent}
                      multiline={true}
                      onChangeText={memoInputHandler}
                      value={enteredMemo}
                      autoFocus
                    />
                  </ScrollView>
                </View>
              </LinearGradient>
            </View>
          </Modal>
        </View>
      )}

      {/* 메모 작성 중 취소 확인 모달 */}
      {isMemoCancelModalVisible &&
        (isUpdateMemoTrue ? (
          <AlertModal
            modalVisible={isMemoCancelModalVisible}
            closeModal={closeMemoCancelModal}
            onConfirm={memoCancelConfirm}
            contentText="메모 수정을 취소하시겠습니까?"
            btnText="확인"
          />
        ) : (
          <AlertModal
            modalVisible={isMemoCancelModalVisible}
            closeModal={closeMemoCancelModal}
            onConfirm={memoCancelConfirm}
            contentText="메모 작성을 취소하시겠습니까?"
            btnText="확인"
          />
        ))}

      {/* 메모가 없을 때, 메모 작성 모달 */}
      {checkMemoLengthZero && (
        <AlertModal
          modalVisible={checkMemoLengthZero}
          closeModal={noLengthModalClose}
          onConfirm={noLengthModalOpenCreate}
          contentText={`물체에 저장된 메모가 없습니다.\n메모를 작성하시겠습니까?`}
          btnText="확인"
        />
      )}

      {/* 튜토리얼 */}
      {isTutorialVisible && page === 1 && (
        <Pressable style={styles.background} onPress={() => setPage(2)}>
          <View style={styles.secondContainer}>
            <Image
              style={styles.second}
              source={require("../../assets/image/tutorial/2.png")}
            />
          </View>
          <View style={styles.btnContainer}>
            <View style={styles.firstContainer}>
              <Image
                style={styles.first}
                source={require("../../assets/image/tutorial/1.png")}
              />
            </View>
            <Image
              source={require("../../assets/image/camerabtn.png")}
              style={styles.addBtn}
            />
          </View>
        </Pressable>
      )}

      {isTutorialVisible && page === 2 && (
        <Pressable style={styles.background} onPress={() => setPage(3)}>
          <View style={styles.thirdContainer}>
            <Image
              style={styles.third}
              source={require("../../assets/image/tutorial/3.png")}
            />
          </View>
        </Pressable>
      )}

      {isTutorialVisible && page === 3 && (
        <Pressable style={styles.background} onPress={() => setPage(4)}>
          <View style={styles.fourthContainer}>
            <Image
              style={styles.fourth}
              source={require("../../assets/image/tutorial/4.png")}
            />
          </View>
        </Pressable>
      )}

      {isTutorialVisible && page === 4 && (
        <Pressable
          style={styles.background}
          onPress={() => {
            setTutorialVisible(false);
            setPage(1);
          }}
        >
          <View style={styles.fifthContainer}>
            <Image
              style={styles.fifth}
              source={require("../../assets/image/tutorial/5.png")}
            />
          </View>
          <View style={styles.startBtnContainer}>
            <ConfirmBtn
              onPress={() => {
                setTutorialVisible(false);
                setPage(1);
              }}
            >
              시작하기
            </ConfirmBtn>
          </View>
        </Pressable>
      )}
    </View>
  );
};

export default MainScreen;

// 라이브러리
import axios from "axios";
import React, { useState, useEffect } from "react";
import { RouteProp } from "@react-navigation/native";
import {
  View,
  Text,
  Image,
  Pressable,
  StyleSheet,
  Modal,
  Alert,
  Dimensions,
  TextInput,
  ScrollView,
} from "react-native";
import { useSelector } from "react-redux";
import LinearGradient from "react-native-linear-gradient";
import { launchImageLibrary } from "react-native-image-picker";

// 타입
import { RootStackParamList } from "../../../App";
import { RootState } from "../../../store/store";
import { GroupData } from "../Group/MyGroupScreen";

// 컴포넌트
import GoBackHeader from "../../../components/Header/GoBackHeader";
import MemoList from "../../../components/Modal/Memo/MemoList";
import MemoDetail, {
  MemoDetailProps,
} from "../../../components/Modal/Memo/MemoDetail";
import LoadingOverlay from "../../../components/LoadingOverlay";
import AlertModal from "../../../components/Modal/AlertModal";

// 스타일
import { styles } from "./MemoStyle";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";

// 통신
import { BACKEND_URL, S3_URL } from "../../../util/http";

// 메뉴메모스크린 타입
type MenuMemoScreenProps = {
  route: RouteProp<RootStackParamList, "MenuMemo">;
};

// 메모 정보 타입
type MemoInfoType = {
  memoSeq: number;
  content: string;
  accessType: string;
  nickname: string;
  updatedAt: string;
  file: string | null;
  itemImage: string | null;
  isBookmarked: boolean;
};

// 태그 시 유저 검색 리스트
type UserData = {
  userSeq: number;
  nickname: string;
  email: string;
}[];

// 모바일 사이즈
const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

// 사진 최대크기 설정
const MAX_WIDTH = calculateDynamicWidth(286);
const MAX_HEIGHT = screenHeight / 2;

const MemuMemo: React.FC<MenuMemoScreenProps> = ({ route }) => {
  // 메뉴 상태값 : saved-북마크 / all-전체 / my-내가 작성한
  const { menuStatus } = route.params;

  // 유저 아이디 (axios 통신용)
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 백엔드에서 주어준 정보
  const [memoData, setMemoData] = useState<MemoInfoType[]>([]);

  // 메모리스트 컴포넌트 상태관리
  const [isMemoList, setIsMemoList] = useState(true);

  // 사진 크기 상태관리
  const [imageWidth, setImageWidth] = useState(MAX_WIDTH);
  const [imageHeight, setImageHeight] = useState(MAX_HEIGHT);

  // 사진첨부 상태관리
  const [uploadedPic, setUploadedPic] = useState<string | null>(null);

  // 이미지 비율 조절 함수
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

  // 메모 정보를 판단하기 위한 AXIOS
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      let response;
      try {
        response = await axios.get(
          BACKEND_URL +
            // 쫀디기
            `/user/${userId}/${
              menuStatus === "saved"
                ? "bookmarks"
                : menuStatus === "all"
                ? "memos"
                : "my-memos"
            }`
        );
        setMemoData(response.data);
      } catch (error) {
        console.log(error);
      } finally {
        setLoading(false);
      }
    };

    if (isMemoList) {
      fetchData();
    }
  }, [isMemoList]);

  // 메모리스트 처리를 위한 더미 함수
  const dummyHandler = () => {
    console.log("히히 더미 함수지롱~~");
  };

  // 사진업로드 함수
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

  // 선택 된 객체 ID 값 저장
  const [pickItem, setPickItem] = useState("");

  // 유저 검색 상태관리
  const [userList, setUserList] = useState<UserData | null>(null);

  // 태그검색 키워드 상태관리
  const [tagSearchText, setTagSearchText] = useState("");

  // 검색결과 상태관리
  const [isSearchResultVisible, setSearchResultVisible] = useState(false);

  // 로딩
  const [loading, setLoading] = useState(false);

  // 메모디테일 컴포넌트 상태관리
  const [isMemoDetail, setIsMemoDetail] = useState(false);

  // 메모수정 컴포넌트 상태관리
  const [isMemoCreate, setIsMemoCreate] = useState(false);

  // 메모아이디 상태관리
  const [memoId, setMemoId] = useState(0);

  // 공개범위 상태관리
  // OPEN: 전체공개 / RESTRICT: 일부공개 / CLOSED: 비공개
  const [openState, setOpenState] = useState("OPEN");

  // 공개범위 선택토글 상태관리
  const [isToggleOpen, setToggleOpen] = useState(false);

  // 이미지 전체보기 상태관리
  const [isFullImageVisible, setFullImageVisible] = useState(false);

  // 메모디테일에서 넘겨준 메모데이터 상태관리
  const [memoDetailData, setMemoDetailData] = useState<MemoDetailProps[]>([]);

  // 메모수정 중 취소 확인 모달 상태관리
  const [isMemoCancelModalVisible, setMemoCancelModalVisible] = useState(false);

  // 내 그룹 리스트 보여주기 상태관리
  const [isMyGroupVisible, setMyGroupVisible] = useState(false);

  // 내 그룹 리스트 상태관리
  const [groupList, setGroupList] = useState<GroupData | null>(null);

  // 태그된 회원 리스트 상태관리
  const [taggedMember, setTaggedMember] = useState<number[]>([]);
  const [taggedMemberList, setTaggedMemberList] = useState<
    { id: number; name: string }[]
  >([]);

  // 태그된 그룹 리스트 상태관리
  const [taggedGroup, setTaggedGroup] = useState<number[]>([]);
  const [taggedGroupList, setTaggedGroupList] = useState<
    { id: number; name: string }[]
  >([]);

  const [enteredMemo, setEnteredMemo] = useState("");

  // 메모리스트 컴포넌트 닫고, 메모디테일 컴포넌트 열기
  const onDetailMemoHandler = (memoSeq: number) => {
    setIsMemoList(false);
    setIsMemoDetail(true);
    setMemoId(memoSeq);
  };

  // 메모디테일 컴포넌트 닫고, 메모리스트 컴포넌트 열기
  const closeDetailOpenList = () => {
    setIsMemoDetail(false);
    setIsMemoList(true);
  };

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

  // 메모디테일 -> 메모 수정
  const updateMemo = (data: MemoDetailProps[]) => {
    setIsMemoDetail(false);
    setIsMemoCreate(true);
    setMemoDetailData(data);
    setOpenState(data[0].accessType);
    memoUpdateTaggedHandler(data);
    setUploadedPic(data[0].file);
    setEnteredMemo(data[0].content);
    setPickItem(data[0].itemName);
  };

  // 메모디테일 -> 메모 삭제
  const deleteMemo = () => {
    setIsMemoDetail(false);
    setIsMemoList(true);
  };

  // 메모수정 종료
  const closeMemoCreate = () => {
    setIsMemoCreate(false);
    setOpenState("OPEN");
    setToggleOpen(false);
  };

  // 전체이미지에서 이미지 삭제
  const deleteUploadedPic = () => {
    setFullImageVisible(false);
  };

  // 메모 취소 모달 확인 버튼 눌렀을 때
  const openMemoCancelModal = () => {
    setMemoCancelModalVisible(true);
  };

  // 메모 취소 모달 취소 버튼 눌렀을 때
  const closeMemoCancelModal = () => {
    setMemoCancelModalVisible(false);
  };

  // 내 그룹을 보여주기
  const openMyGroupList = () => {
    setMyGroupVisible(true);
    getGroupList();
  };

  const getGroupList = async () => {
    try {
      const res = await axios({
        method: "GET",
        url: BACKEND_URL + `/user/${userId}/my-teams`,
        // url: BACKEND_URL + `/user/30/my-teams`, // 쫀듸기
      });
      setGroupList(res.data);
    } catch (err) {
      console.log(err);
    }
  };

  // 태그 검색 닫기
  const closeTagSearch = () => {
    setSearchResultVisible(false);
    setTagSearchText("");
    setMyGroupVisible(false);
  };

  // 그룹 태그하기
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

  // 공개범위설정 토글 열기
  const selectOpenState = () => {
    setToggleOpen(!isToggleOpen);
  };

  // 공개범위설정
  const chooseOpenState = (state: string) => {
    setOpenState(state);
    setToggleOpen(false);
  };

  // 오늘 날짜 가져오기
  function getFormattedDate(): string {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0"); // 월이 0부터 시작하므로 1을 더해줌
    const day = String(date.getDate()).padStart(2, "0");

    return `${year}. ${month}. ${day}.`;
  }

  // 날짜 형태 변경
  const currentDate = getFormattedDate();

  // 이미지 전체 보기
  const openFullImage = () => {
    setFullImageVisible(true);
  };

  // 전체 이미지 닫기
  const closeFullImage = () => {
    setFullImageVisible(false);
  };

  const MemoCreate = async () => {
    if (!enteredMemo) {
      Alert.alert("내용을 입력해 주세요!");
    } else {
      await axios({
        method: "PUT",
        url: BACKEND_URL + `/memos/${memoId}`,
        // headers: {
        //   "Content-Type": "application/json",
        //   Authorization: "Bearer " + token,
        // },
        data: {
          content: enteredMemo,
          accessType: openState,
          userId: userId,
          // userId: 30, // 쫀듸기
          newFile: uploadedPic,
          itemName: pickItem,
          // itemName: "8ef97a8a0be", // 쫀듸기
          taggedUserList: taggedMember,
          taggedTeamList: taggedGroup,
        },
      })
        .then((res) => {
          if (res.request.status === 200) {
            console.log("메모 수정 성공");
            //물체 표시 생성
            setTaggedMember([]);
            setTaggedGroup([]);
            setTaggedMemberList([]);
            setTaggedGroupList([]);
          }
        })
        .catch((err) => {
          console.log(err);
        });
    }
  };

  // 메모 수정 함수
  const memoConfirmHandler = () => {
    setIsMemoCreate(false);
    setEnteredMemo("");
    setOpenState("OPEN");
    setIsMemoList(true);
    MemoCreate();
  };

  // 메모 작성 함수
  const memoInputHandler = (enteredText: string) => {
    setEnteredMemo(enteredText);
  };

  // 메모 취소 모달의 확인 버튼 눌렀을 때
  const memoCancelConfirm = () => {
    setEnteredMemo("");
    setOpenState("OPEN");
    setToggleOpen(false);
    setMemoCancelModalVisible(false);
    setTagSearchText("");
    setTaggedMember([]);
    setMyGroupVisible(false);
    setTaggedMember([]);
    setTaggedGroup([]);
    setTaggedMemberList([]);
    setTaggedGroupList([]);
    setIsMemoCreate(false);
    setIsMemoDetail(true);
  };

  const dummyFunction = (checker: boolean) => {
    console.log("더미데이터 : " + checker);
  };

  return (
    <>
      {loading ? (
        <LoadingOverlay />
      ) : (
        <>
          {memoData.length > 0 ? (
            // 메모 있을 때..
            <LinearGradient
              colors={["#F5F5F5", "#E9E9E9"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 0, y: 1 }}
              style={{ flex: 1 }}
            >
              {isMemoList && (
                <>
                  <GoBackHeader />
                  <View>
                    {menuStatus === "saved" && (
                      <Text style={styles.title}>저장된 메모</Text>
                    )}
                    {menuStatus === "all" && (
                      <Text style={styles.title}>전체 메모</Text>
                    )}
                    {menuStatus === "my" && (
                      <Text style={styles.title}>내 메모</Text>
                    )}
                  </View>
                  <MemoList
                    onMemoWritePress={dummyHandler}
                    onMemoDetailPress={onDetailMemoHandler}
                    id={null}
                    memoStatus={menuStatus}
                    memoListStatus={isMemoList}
                    onMemoZeroLength={dummyFunction}
                  />
                </>
              )}
              {isMemoDetail && (
                <>
                  <GoBackHeader />
                  <Pressable
                    onPress={closeDetailOpenList}
                    style={menumemostyles.back}
                  />
                  <MemoDetail
                    memoSeq={memoId}
                    onMemoUpdatePress={updateMemo}
                    onMemoDeletePress={deleteMemo}
                  />
                </>
              )}
              {isMemoCreate && (
                <>
                  <View style={[menumemostyles.background, { zIndex: 2 }]}>
                    <Modal
                      transparent={true}
                      animationType="fade"
                      visible={isMemoCreate}
                      onRequestClose={closeMemoCreate}
                    >
                      {/* 첨부 사진 상세 조회 */}
                      {isFullImageVisible && (
                        <>
                          <Pressable
                            style={[
                              menumemostyles.uploadedImgBg,
                              {
                                backgroundColor: "rgba(0, 0, 0, 0.5)",
                                zIndex: 2,
                              },
                            ]}
                            onPress={() => setFullImageVisible(false)}
                          />
                          {/* 첨부 사진 삭제 버튼 */}
                          <Pressable
                            onPress={deleteUploadedPic}
                            style={[
                              menumemostyles.binContainer,
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
                                      imageWidth / 2 -
                                      calculateDynamicWidth(20),
                                  },
                                ],
                              },
                            ]}
                          >
                            <Image
                              source={require("../../../assets/icons/bin.png")}
                              style={menumemostyles.bin}
                            />
                          </Pressable>
                          {/* 첨부 사진 */}
                          {uploadedPic && (
                            <Image
                              source={{ uri: uploadedPic }}
                              style={[
                                menumemostyles.uploadedFullImg,
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
                                style={menumemostyles.closeTagSearch}
                                onPress={closeTagSearch}
                              />
                              <View style={menumemostyles.tagResultContainer}>
                                <View
                                  style={[
                                    menumemostyles.tagSearchContainer,
                                    {
                                      borderBottomWidth: 1,
                                      borderBottomColor:
                                        "rgba(44, 44, 44, 0.5)",
                                    },
                                  ]}
                                >
                                  <Text style={menumemostyles.tagText}>@</Text>
                                  <TextInput
                                    style={menumemostyles.tagText}
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
                                    style={
                                      menumemostyles.tagResultInnerContainer
                                    }
                                    onPress={() =>
                                      addTaggedGroup(
                                        group.teamSeq,
                                        group.teamName
                                      )
                                    }
                                    key={idx}
                                  >
                                    <Text style={menumemostyles.tagText}>
                                      {group.teamName}{" "}
                                      <Text style={menumemostyles.email}>
                                        {" "}
                                        #{group.teamSeq}
                                      </Text>
                                    </Text>
                                  </Pressable>
                                ))}
                              </View>
                            </>
                          ) : (
                            <View style={menumemostyles.tagContainer}>
                              <Pressable
                                onPress={openMyGroupList}
                                style={menumemostyles.tagSearchContainer}
                              >
                                <Text style={menumemostyles.tagText}>
                                  @{" "}
                                  <Text
                                    style={{ color: "rgba(44, 44, 44, 0.5)" }}
                                  >
                                    태그할 그룹을 선택하거나 유저를 검색해
                                    주세요
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
                          <ScrollView
                            horizontal
                            style={menumemostyles.tagResultBox}
                          >
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
                                  style={menumemostyles.taggedMemberContainer}
                                >
                                  <Text style={menumemostyles.tagText}>
                                    @ {group.name}
                                  </Text>
                                  <Pressable
                                    onPress={() => removeTaggedGroup(group.id)}
                                  >
                                    <Image
                                      source={require("../../../assets/icons/cancel_sm.png")}
                                      style={menumemostyles.cancelIcon}
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
                                  style={menumemostyles.taggedMemberContainer}
                                >
                                  <Text style={menumemostyles.tagText}>
                                    @ {member.name}
                                  </Text>
                                  <Pressable
                                    onPress={() =>
                                      removeTaggedMember(member.id)
                                    }
                                  >
                                    <Image
                                      source={require("../../../assets/icons/cancel_sm.png")}
                                      style={menumemostyles.cancelIcon}
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
                            style={menumemostyles.closeTagSearch}
                            onPress={closeTagSearch}
                          />
                          <View style={menumemostyles.tagResultContainer}>
                            <View
                              style={[
                                menumemostyles.tagSearchContainer,
                                {
                                  borderBottomWidth: 1,
                                  borderBottomColor: "rgba(44, 44, 44, 0.5)",
                                },
                              ]}
                            >
                              <Text style={menumemostyles.tagText}>@</Text>
                              <TextInput
                                style={menumemostyles.tagText}
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
                                  style={menumemostyles.tagResultInnerContainer}
                                  onPress={() =>
                                    addTaggedMember(user.userSeq, user.nickname)
                                  }
                                  key={idx}
                                >
                                  <Text style={menumemostyles.tagText}>
                                    {user.nickname}{" "}
                                    <Text style={menumemostyles.email}>
                                      {" "}
                                      {user.email}
                                    </Text>
                                  </Text>
                                </Pressable>
                              ))
                            ) : (
                              <View
                                style={menumemostyles.tagResultInnerContainer}
                              >
                                <Text style={menumemostyles.tagText}>
                                  검색 결과가 없습니다.
                                </Text>
                              </View>
                            )}
                          </View>
                        </>
                      )}
                      <View style={menumemostyles.memoContainer}>
                        <LinearGradient
                          colors={["#FFFFFF", "#F5F5F5"]}
                          start={{ x: 0, y: 0 }}
                          end={{ x: 0, y: 1 }}
                          style={{ flex: 1 }}
                        >
                          <View style={{ padding: calculateDynamicWidth(15) }}>
                            {/* 공개 범위 설정 버튼 */}
                            <View style={menumemostyles.memoInnerContainer}>
                              <View>
                                {openState === "OPEN" && (
                                  <Pressable onPress={selectOpenState}>
                                    <Image
                                      source={require("../../../assets/image/public.png")}
                                      style={menumemostyles.openState}
                                    />
                                  </Pressable>
                                )}
                                {openState === "RESTRICT" && (
                                  <Pressable onPress={selectOpenState}>
                                    <Image
                                      source={require("../../../assets/image/restrict.png")}
                                      style={menumemostyles.openState}
                                    />
                                  </Pressable>
                                )}
                                {openState === "CLOSED" && (
                                  <Pressable onPress={selectOpenState}>
                                    <Image
                                      source={require("../../../assets/image/closed.png")}
                                      style={menumemostyles.openState}
                                    />
                                  </Pressable>
                                )}
                                {isToggleOpen && (
                                  <View style={menumemostyles.toggleContainer}>
                                    <Pressable
                                      onPress={() => chooseOpenState("OPEN")}
                                    >
                                      <View
                                        style={
                                          menumemostyles.toggleContentContainer
                                        }
                                      >
                                        <Text style={menumemostyles.toggleText}>
                                          전체공개
                                        </Text>
                                        {openState === "OPEN" && (
                                          <View
                                            style={
                                              menumemostyles.blueDotContainer
                                            }
                                          >
                                            <View
                                              style={menumemostyles.blueDot}
                                            ></View>
                                          </View>
                                        )}
                                      </View>
                                    </Pressable>
                                    <Pressable
                                      onPress={() =>
                                        chooseOpenState("RESTRICT")
                                      }
                                    >
                                      <View
                                        style={
                                          menumemostyles.toggleContentContainer
                                        }
                                      >
                                        <Text style={menumemostyles.toggleText}>
                                          일부공개
                                        </Text>
                                        {openState === "RESTRICT" && (
                                          <View
                                            style={
                                              menumemostyles.blueDotContainer
                                            }
                                          >
                                            <View
                                              style={menumemostyles.blueDot}
                                            ></View>
                                          </View>
                                        )}
                                      </View>
                                    </Pressable>
                                    <Pressable
                                      onPress={() => chooseOpenState("CLOSED")}
                                    >
                                      <View
                                        style={[
                                          menumemostyles.toggleClosedContentContainer,
                                          {
                                            paddingLeft:
                                              calculateDynamicWidth(16),
                                          },
                                        ]}
                                      >
                                        <Text style={menumemostyles.toggleText}>
                                          비공개
                                        </Text>
                                        {openState === "CLOSED" && (
                                          <View
                                            style={
                                              menumemostyles.blueDotContainer
                                            }
                                          >
                                            <View
                                              style={[
                                                menumemostyles.blueDot,
                                                {
                                                  backgroundColor: Colors.text,
                                                },
                                              ]}
                                            ></View>
                                          </View>
                                        )}
                                      </View>
                                    </Pressable>
                                  </View>
                                )}
                                <Text style={menumemostyles.currentDate}>
                                  {currentDate}
                                </Text>
                              </View>
                              <View
                                style={menumemostyles.memoInnerBtnContainer}
                              >
                                <Pressable onPress={selectImageHanlder}>
                                  <Image
                                    source={require("../../../assets/icons/addpic.png")}
                                    style={menumemostyles.addPic}
                                  />
                                </Pressable>
                                <Pressable onPress={memoConfirmHandler}>
                                  <View>
                                    <Image
                                      source={require("../../../assets/icons/confirm.png")}
                                      style={menumemostyles.confirm}
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
                                    style={menumemostyles.uploadedImg}
                                  />
                                </Pressable>
                              )}
                              <TextInput
                                style={menumemostyles.memoContent}
                                multiline={true}
                                onChangeText={memoInputHandler}
                                value={enteredMemo}
                              />
                            </ScrollView>
                          </View>
                        </LinearGradient>
                      </View>
                    </Modal>
                  </View>

                  {/* 메모 수정 중 취소 확인 모달 */}
                  {isMemoCancelModalVisible && (
                    <AlertModal
                      modalVisible={isMemoCancelModalVisible}
                      closeModal={closeMemoCancelModal}
                      onConfirm={memoCancelConfirm}
                      contentText="메모 수정을 취소하시겠습니까?"
                      btnText="확인"
                    />
                  )}
                </>
              )}
            </LinearGradient>
          ) : (
            // 메모 없을 때..
            <LinearGradient
              colors={["#F5F5F5", "#E9E9E9"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 0, y: 1 }}
              style={{ flex: 1 }}
            >
              <GoBackHeader />
              {menuStatus === "saved" && (
                <View>
                  <Text style={styles.title}>저장된 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>저장된 메모가 없습니다</Text>
                  </View>
                </View>
              )}
              {menuStatus === "all" && (
                <View>
                  <Text style={styles.title}>전체 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>메모가 없습니다</Text>
                  </View>
                </View>
              )}
              {menuStatus === "my" && (
                <View>
                  <Text style={styles.title}>내 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>작성한 메모가 없습니다</Text>
                  </View>
                </View>
              )}
            </LinearGradient>
          )}
        </>
      )}
    </>
  );
};

export default MemuMemo;

const menumemostyles = StyleSheet.create({
  back: {
    backgroundColor: "transparent",
    position: "absolute",
    width: 15,
    height: 25,
    marginLeft: 15,
    left: 5,
    top: 36,
    padding: 10,
  },
  background: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
  },
  uploadedImgBg: {
    flex: 1,
    backgroundColor: "transparent",
    zIndex: 1,
    marginBottom: -screenHeight,
  },
  binContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    zIndex: 3,
  },
  bin: {
    width: calculateDynamicWidth(15.61),
    height: calculateDynamicWidth(16),
  },
  uploadedFullImg: {
    zIndex: 2,
    borderRadius: calculateDynamicWidth(10),
    position: "absolute",
    top: "50%",
    left: "50%",
  },
  closeTagSearch: {
    flex: 1,
    backgroundColor: "transparent",
    zIndex: 1,
    marginTop: -screenHeight,
  },
  tagResultContainer: {
    width: calculateDynamicWidth(306),
    maxHeight: calculateDynamicWidth(356),
    borderRadius: calculateDynamicWidth(15),
    backgroundColor: "#ECECEC",
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      {
        translateY: -(
          calculateDynamicWidth(306) / 2 +
          calculateDynamicWidth(50)
        ),
      },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
    zIndex: 1,
    elevation: 4,
    paddingHorizontal: calculateDynamicWidth(10),
    justifyContent: "center",
  },
  tagSearchContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  tagText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: Colors.text,
  },
  tagResultInnerContainer: {
    marginVertical: calculateDynamicWidth(12),
  },
  email: {
    fontSize: calculateDynamicWidth(12),
    color: "rgba(44, 44, 44, 0.5)",
  },
  tagContainer: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(50),
    borderRadius: calculateDynamicWidth(15),
    backgroundColor: "#ECECEC",
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      {
        translateY: -(
          calculateDynamicWidth(306) / 2 +
          calculateDynamicWidth(50)
        ),
      },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
    zIndex: 1,
    elevation: 4,
    paddingHorizontal: calculateDynamicWidth(10),
    justifyContent: "center",
  },
  tagResultBox: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(50),
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      {
        translateY: -(
          calculateDynamicWidth(306) / 2 +
          calculateDynamicWidth(100)
        ),
      },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
    zIndex: 1,
    paddingHorizontal: calculateDynamicWidth(10),
  },
  taggedMemberContainer: {
    flexDirection: "row",
    alignItems: "center",
    elevation: 4,
    height: calculateDynamicWidth(26),
    borderRadius: calculateDynamicWidth(50),
    paddingHorizontal: calculateDynamicWidth(8),
    marginRight: calculateDynamicWidth(10),
  },
  cancelIcon: {
    width: calculateDynamicWidth(10),
    height: calculateDynamicWidth(10),
    marginLeft: calculateDynamicWidth(5),
  },
  memoContainer: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(306),
    maxHeight: calculateDynamicWidth(306),
    borderRadius: calculateDynamicWidth(15),
    overflow: "scroll",
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(306) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },
  memoInnerContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  openState: {
    width: calculateDynamicWidth(81),
    height: calculateDynamicWidth(23),
  },
  blueDotContainer: {
    marginLeft: calculateDynamicWidth(3),
    position: "absolute",
    top: calculateDynamicWidth(23) / 2,
    right: calculateDynamicWidth(7),
    transform: [{ translateY: -calculateDynamicWidth(8) / 2 }],
  },
  blueDot: {
    width: calculateDynamicWidth(8),
    height: calculateDynamicWidth(8),
    borderRadius: calculateDynamicWidth(8) / 2,
    backgroundColor: Colors.blue500,
  },
  toggleContainer: {
    width: calculateDynamicWidth(81),
    height: calculateDynamicWidth(69),
    borderRadius: calculateDynamicWidth(8),
    backgroundColor: "white",
    elevation: 4,
    alignItems: "center",
    position: "absolute",
    top: calculateDynamicWidth(23),
    zIndex: 1,
  },
  toggleContentContainer: {
    width: calculateDynamicWidth(81),
    flexDirection: "row",
    borderBottomWidth: 0.8,
    borderBlockColor: Colors.hover,
    paddingBottom: calculateDynamicWidth(2.5),
    paddingLeft: calculateDynamicWidth(10.5),
  },
  toggleClosedContentContainer: {
    width: calculateDynamicWidth(81),
    flexDirection: "row",
    paddingLeft: calculateDynamicWidth(10.5),
  },
  toggleText: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(14),
    color: Colors.text,
  },
  currentDate: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: Colors.hover,
    marginTop: calculateDynamicWidth(4),
  },
  memoInnerBtnContainer: {
    flexDirection: "row",
  },
  addPic: {
    width: calculateDynamicWidth(21),
    height: calculateDynamicWidth(20),
    marginRight: calculateDynamicWidth(10),
  },
  confirm: {
    width: calculateDynamicWidth(21),
    height: calculateDynamicWidth(15.21),
  },
  uploadedImg: {
    width: calculateDynamicWidth(257),
    height: calculateDynamicWidth(106),
    borderRadius: calculateDynamicWidth(10),
    marginTop: calculateDynamicWidth(5),
  },
  memoContent: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
  },
});

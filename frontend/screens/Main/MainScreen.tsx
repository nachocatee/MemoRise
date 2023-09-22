import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Modal,
  Pressable,
  Image,
  ActivityIndicator,
  StyleSheet,
  ScrollView,
  Dimensions,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { BlurView } from "@react-native-community/blur";
import { Camera, useCameraDevices } from "react-native-vision-camera";
import { useIsFocused } from "@react-navigation/native";
import { launchImageLibrary } from "react-native-image-picker";
import axios from "axios";

import MainHeader from "../../components/Header/MainHeader";
import { styles } from "./MainStyle";
import { TextInput } from "react-native-gesture-handler";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import Colors from "../../constants/colors";
import MemoBtnModal from "../../components/Modal/Memo/MemoBtnModal";
import AlertModal from "../../components/Modal/AlertModal";
import { BACKEND_URL, S3_URL } from "../../util/http";

const screenHeight = Dimensions.get("window").height;

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
          .post(BACKEND_URL + "/user/upload", formData, {
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
      // 원본 이미지의 크기를 가져옵니다.
      Image.getSize(uploadedPic, (width, height) => {
        // 원본 이미지의 비율을 계산합니다.
        const aspectRatio = width / height;

        // 비율을 유지하면서 크기를 조절합니다.
        if (width > height) {
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

  // 메모 작성 내용
  const [memoContent, setMemoContent] = useState("");
  const [enteredMemo, setEnteredMemo] = useState("");
  // 메모 조회 상태관리
  const [memoItemVisible, setMemoItemVisible] = useState(true);

  // 메모모달 조회 함수
  // 나중에 객체 탐지해서 메모 개수 나오면 함수 적용
  const checkMemoHandler = () => {
    setMemoItemVisible(true);
  };

  const memoInputHandler = (enteredText: string) => {
    setEnteredMemo(enteredText);
  };

  const memoConfirmHandler = () => {
    setMemoContent(enteredMemo);
    setMemoCreateModalVisible(false);
  };

  // 공개 범위 설정
  // 0: 전체공개, 1: 일부공개, 2: 비공개
  const [openState, setOpenState] = useState(0);
  const [isToggleOpen, setToggleOpen] = useState(false);

  const selectOpenState = () => {
    setToggleOpen(!isToggleOpen);
  };

  const chooseOpenState = (state: number) => {
    setOpenState(state);
    setToggleOpen(false);
  };

  // 메모 작성 모달
  const [isMemoCreateModalVisible, setMemoCreateModalVisible] = useState(false);

  const closeMemoCreateModal = () => {
    setMemoCreateModalVisible(false);
    setOpenState(0);
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
    setOpenState(0);
    setToggleOpen(false);
    setMemoCancelModalVisible(false);
    setMemoCreateModalVisible(false);
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

  // 카메라 로직
  // 연결된 디바이스 확인
  const devices = useCameraDevices();

  // 후방 카메라
  const device = devices.back;

  // 페이지 첫 렌더링 시 허용 권한 체크
  useEffect(() => {
    checkPermission();
  }, []);

  // 카메라 허용 권한 확인
  const checkPermission = async () => {
    const cameraPermission = await Camera.getCameraPermissionStatus(); // 현재 카메라 권한 상태
    if (cameraPermission === "denied") {
      return Camera.requestCameraPermission(); // 카메라 허용 요청
    }
  };

  if (device == null) return <ActivityIndicator />; // 디바이스가 없을 시 원형 로딩 표시기를 표시

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
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isFocused}
          photo
        />
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
      {memoItemVisible && <MemoItem />}

      {/* 알림 모달 */}
      {isNotificationModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
          blurType="dark"
        >
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
        </BlurView>
      )}

      {/* 메모 버튼 모달 */}
      {isMemoBtnModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 2,
          }}
          blurType="dark"
        >
          <Modal
            transparent={true}
            animationType="fade"
            visible={isMemoBtnModalVisible}
            onRequestClose={closeMemoBtnModal}
          >
            <MemoBtnModal
              openMemoCreateModal={openMemoCreateModal}
              closeModal={closeMemoBtnModal}
            />
          </Modal>
        </BlurView>
      )}

      {/* 메모 작성 모달 */}
      {isMemoCreateModalVisible && (
        <BlurView
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 2,
          }}
          blurType="dark"
        >
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
            {openState === 1 && !isSearchResultVisible && (
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
            {openState === 1 && taggedMember[0] && (
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
                      {openState === 0 && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/public.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {openState === 1 && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/restrict.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {openState === 2 && (
                        <Pressable onPress={selectOpenState}>
                          <Image
                            source={require("../../assets/image/closed.png")}
                            style={styles.openState}
                          />
                        </Pressable>
                      )}
                      {isToggleOpen && (
                        <View style={styles.toggleContainer}>
                          <Pressable onPress={() => chooseOpenState(0)}>
                            <View style={styles.toggleContentContainer}>
                              <Text style={styles.toggleText}>전체공개</Text>
                              {openState === 0 && (
                                <View style={styles.blueDotContainer}>
                                  <View style={styles.blueDot}></View>
                                </View>
                              )}
                            </View>
                          </Pressable>
                          <Pressable onPress={() => chooseOpenState(1)}>
                            <View style={styles.toggleContentContainer}>
                              <Text style={styles.toggleText}>일부공개</Text>
                              {openState === 1 && (
                                <View style={styles.blueDotContainer}>
                                  <View style={styles.blueDot}></View>
                                </View>
                              )}
                            </View>
                          </Pressable>
                          <Pressable onPress={() => chooseOpenState(2)}>
                            <View
                              style={[
                                styles.toggleClosedContentContainer,
                                { paddingLeft: calculateDynamicWidth(16) },
                              ]}
                            >
                              <Text style={styles.toggleText}>비공개</Text>
                              {openState === 2 && (
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
                    />
                  </ScrollView>
                </View>
              </LinearGradient>
            </View>
          </Modal>
        </BlurView>
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

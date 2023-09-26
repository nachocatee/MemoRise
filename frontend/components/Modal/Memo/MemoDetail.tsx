// 라이브러리
import axios from "axios";
import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
  Dimensions,
  Modal,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";

// 컴포넌트
import AlertModal from "../AlertModal";
import BookMarkBtn from "../../Button/BookMarkBtn";

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";
import { styles } from "../../../screens/Main/MainStyle";

// 통신
import { BACKEND_URL } from "../../../util/http";

// 스크린 높이
const screenHeight = Dimensions.get("window").height;

// 이미지 최대 크기 설정
const MAX_WIDTH = calculateDynamicWidth(286);
const MAX_HEIGHT = screenHeight / 2;

// 타입관리
export type MemoDetailProps = {
  accessType: string;
  content: string;
  file: string;
  isBookmarked: boolean;
  itemImage: string;
  itemSeq: number;
  nickname: string;
  taggedUserList: [
    {
      nickname: string;
    }
  ];
  updatedAt: string;
};

// 메인페이지 상태관리를 위한 타입 지정
type MemoDetailProp = {
  memoSeq: number | null;
  onMemoUpdatePress: (data: MemoDetailProps[]) => void;
  onMemoDeletePress: () => void;
};

const MemoDetail: React.FC<MemoDetailProp> = ({
  memoSeq,
  onMemoUpdatePress,
  onMemoDeletePress,
}) => {
  // 메모 상세 조회 상태관리
  const [memoDetailData, setMemoDetailData] = useState<MemoDetailProps[]>([]);

  // 메모 상세 조회 날짜 상태관리
  const [memoDetailCalendar, setMemoDetailCalendar] = useState("");

  // 날짜 변경 함수
  const formatData = (inputData: string) => {
    const date = new Date(inputData);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = date.getHours();
    const minutes = String(date.getMinutes()).padStart(2, "0");

    const ampm = hours >= 12 ? "오후" : "오전";
    const formattedHours = hours > 12 ? hours - 12 : hours;
    const formattedHoursString = String(formattedHours).padStart(2, "0");

    return `${year}. ${month}. ${day}. ${ampm} ${formattedHoursString}:${minutes}`;
  };

  // 메모 상세 조회를 위한 AXIOS
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get(BACKEND_URL + `/memos/${memoSeq}/23`);
        setMemoDetailData([res.data]);
        setMemoPic(res.data.file);
        setMemoDetailCalendar(formatData(res.data.updatedAt));
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

  // 이미지 비율 축소를 위한 상태관리
  const [memoPic, setMemoPic] = useState("");
  const [imageWidth, setImageWidth] = useState(MAX_WIDTH);
  const [imageHeight, setImageHeight] = useState(MAX_HEIGHT);

  // 이미지 모달 상태관리
  const [isFullImageVisible, setIsFullImageVisible] = useState(false);

  // 원본 이미지 보기 함수
  const openFullImage = () => {
    setIsFullImageVisible(true);
    setMemoPic(memoDetailData[0].file);
  };

  // 이미지 모달 상태 변경 함수
  const closeFullImage = () => {
    setIsFullImageVisible(false);
  };

  // 이미지를 비율에 맞게 축소
  useEffect(() => {
    if (memoPic) {
      // 원본 이미지 크기
      Image.getSize(memoPic, (width, height) => {
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
  }, [memoPic]);

  // 메모 이미지 삭제 함수
  const deleteImage = () => {
    setMemoPic("");
    setIsFullImageVisible(false);
  };

  // 메모 삭제 모달 상태관리
  const [isDeleteMemoModalVisible, setIsDeleteMemoModalVisible] =
    useState(false);

  // 메모 삭제 물어보는 함수
  const deleteMemoHandler = () => {
    setIsDeleteMemoModalVisible(true);
  };

  // 메모 삭제 모달 종료 함수
  const closeDeleteMemoModal = () => {
    setIsDeleteMemoModalVisible(false);
  };

  // 메모 삭제
  const memoDeleteConfirm = () => {
    onMemoDeletePress();
    setIsDeleteMemoModalVisible(false);
    axios.delete(BACKEND_URL + `/memos/${memoSeq}/`).catch((error) => {
      console.log(error);
    });
  };

  return (
    <>
      <View style={detailStyle.mainContainer}>
        <LinearGradient
          colors={["#FFFFFF", "#F5F5F5"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={detailStyle.memoContainer}
        >
          <View style={detailStyle.innerContainer}>
            <View>
              <View style={detailStyle.rowSpaceBetween}>
                {memoDetailData[0] && (
                  <Text style={detailStyle.calendar}>{memoDetailCalendar}</Text>
                )}
                <View style={detailStyle.iconContainer}>
                  <Pressable onPress={() => onMemoUpdatePress(memoDetailData)}>
                    <Image
                      source={require("../../../assets/icons/update.png")}
                      style={detailStyle.icon}
                    />
                  </Pressable>
                  <Pressable onPress={deleteMemoHandler}>
                    <Image
                      source={require("../../../assets/icons/delete.png")}
                      style={detailStyle.icon}
                    />
                  </Pressable>
                </View>
              </View>
              <View style={detailStyle.rowSpaceBetween2}>
                {memoDetailData[0] && (
                  <Text style={detailStyle.nickname}>
                    {memoDetailData[0].nickname}
                  </Text>
                )}
                {memoDetailData[0] && (
                  <Text style={detailStyle.open}>
                    {memoDetailData[0].accessType}
                  </Text>
                )}
              </View>
            </View>
            <ScrollView style={detailStyle.contentContainer}>
              {memoPic !== "" ? (
                <>
                  <Pressable onPress={openFullImage}>
                    <Image
                      source={{
                        uri: memoPic,
                      }}
                      style={detailStyle.photo}
                    />
                  </Pressable>
                  {memoDetailData[0] && (
                    <Text style={detailStyle.content}>
                      {memoDetailData[0].content}
                    </Text>
                  )}
                </>
              ) : (
                <>
                  {memoDetailData[0] && (
                    <Text style={detailStyle.content}>
                      {memoDetailData[0].content}
                    </Text>
                  )}
                </>
              )}
            </ScrollView>
          </View>
        </LinearGradient>
        <BookMarkBtn
          memoSeq={memoSeq}
          detailStyle={[detailStyle.bookmark, detailStyle.bookmarkSize]}
          // 하드코딩 : false부분을 isbookmarked로 바꿀 것!
          bookmarkType={memoDetailData[0] && memoDetailData[0].isBookmarked}
        />
      </View>
      {isFullImageVisible && (
        <View style={[styles.background, { zIndex: 2 }]}>
          <Modal
            transparent={true}
            animationType="fade"
            visible={isFullImageVisible}
            onRequestClose={closeFullImage}
          >
            <Pressable
              style={[
                styles.uploadedImgBg,
                { backgroundColor: "rgba(0, 0, 0, 0.5)", zIndex: 2 },
              ]}
              onPress={closeFullImage}
            />
            {/* 첨부 사진 삭제 버튼 */}
            <Pressable
              onPress={deleteImage}
              style={[
                styles.binContainer,
                {
                  transform: [
                    {
                      translateY: -(
                        imageHeight / 2 +
                        calculateDynamicWidth(25)
                      ),
                    },
                    {
                      translateX: imageWidth / 2 - calculateDynamicWidth(20),
                    },
                  ],
                },
              ]}
            >
              <Image
                source={require("../../../assets/icons/bin.png")}
                style={styles.bin}
              />
            </Pressable>
            {/* 첨부 사진 */}
            <Image
              source={{ uri: memoPic }}
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
          </Modal>
        </View>
      )}
      {isDeleteMemoModalVisible && (
        <AlertModal
          modalVisible={isDeleteMemoModalVisible}
          closeModal={closeDeleteMemoModal}
          onConfirm={memoDeleteConfirm}
          contentText="정말 삭제하시겠습니까?"
          btnText="삭제"
        />
      )}
    </>
  );
};

export default MemoDetail;

const detailStyle = StyleSheet.create({
  mainContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(306) / 2 },
      { translateX: -calculateDynamicWidth(306) / 2 },
    ],
  },

  memoContainer: {
    width: calculateDynamicWidth(306),
    minHeight: calculateDynamicWidth(306),
    borderRadius: calculateDynamicWidth(15),
  },

  innerContainer: {
    margin: calculateDynamicWidth(9),
    flex: 1,
  },

  rowSpaceBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    marginBottom: 3,
  },

  rowSpaceBetween2: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 3,
  },

  calendar: {
    color: Colors.hover,
    fontFamily: "Pretendard-Regular",
  },

  iconContainer: {
    flexDirection: "row",
  },

  icon: {
    width: calculateDynamicWidth(14),
    height: calculateDynamicWidth(14),
    marginLeft: 10,
  },

  nickname: {
    color: Colors.blue500,
    fontFamily: "Pretendard-Medium",
  },

  open: {
    color: "rgba(76, 106, 255, 0.6)",
    fontFamily: "Pretendard-Medium",
  },

  contentContainer: {
    maxHeight: calculateDynamicWidth(350),
  },

  photo: {
    width: calculateDynamicWidth(275),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(10),
    marginLeft: calculateDynamicWidth(5),
  },

  content: {
    color: Colors.text,
    fontSize: 18,
    fontFamily: "Pretendard-Regular",
    marginVertical: 5,
    marginHorizontal: 10,
  },

  bookmark: {
    position: "absolute",
    top: 0,
    marginLeft: calculateDynamicWidth(16),
  },

  bookmarkSize: {
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(23),
  },
});

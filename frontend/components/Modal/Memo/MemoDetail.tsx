// 라이브러리
import { useState, useEffect } from "react";
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

// 스타일
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import Colors from "../../../constants/colors";
import { styles } from "../../../screens/Main/MainStyle";

// 스크린 높이
const screenHeight = Dimensions.get("window").height;

// 이미지 최대 크기 설정
const MAX_WIDTH = calculateDynamicWidth(286);
const MAX_HEIGHT = screenHeight / 2;

const MemoDetail = () => {
  // 이미지 비율 축소를 위한 상태관리
  const [memoPic, setMemoPic] = useState(
    "https://b106-memorise.s3.ap-northeast-2.amazonaws.com/profile-image/dc971e2c-4a4a-4330-9dfe-ea691ad9bcdf.png"
  );
  const [imageWidth, setImageWidth] = useState(MAX_WIDTH);
  const [imageHeight, setImageHeight] = useState(MAX_HEIGHT);

  // 이미지 모달 상태관리
  const [isFullImageVisible, setIsFullImageVisible] = useState(false);

  // 원본 이미지 보기 함수
  const openFullImage = () => {
    setIsFullImageVisible(true);
    setMemoPic(
      "https://b106-memorise.s3.ap-northeast-2.amazonaws.com/profile-image/dc971e2c-4a4a-4330-9dfe-ea691ad9bcdf.png"
    );
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

  // 북마크 체크 여부 파악을 위한 상태관리
  const [isBookMark, setIsBookMark] = useState(false);

  // 북마크 체크에 따른 변경 함수
  const changeIsBookMark = () => {
    setIsBookMark(!isBookMark);
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
                <Text style={detailStyle.calendar}>
                  2023. 09. 24 오전 12:50
                </Text>
                <View style={detailStyle.iconContainer}>
                  <Pressable>
                    <Image
                      source={require("../../../assets/icons/update.png")}
                      style={detailStyle.icon}
                    />
                  </Pressable>
                  <Pressable>
                    <Image
                      source={require("../../../assets/icons/delete.png")}
                      style={detailStyle.icon}
                    />
                  </Pressable>
                </View>
              </View>
              <View style={detailStyle.rowSpaceBetween2}>
                <Text style={detailStyle.nickname}>김준형</Text>
                <Text style={detailStyle.open}>전체공개</Text>
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
                  <Text style={detailStyle.content}>
                    Hate to give the satisfaction, asking how you're doing now.
                    How's the castle built off people you pretend to care about?
                    Just what you wantedLook at you, cool guy, you got it I see
                    the parties and the
                  </Text>
                </>
              ) : (
                <Text style={detailStyle.content}>
                  Hate to give the satisfaction, asking how you're doing now.
                  How's the castle built off people you pretend to care about?
                  Just what you wantedLook at you, cool guy, you got it I see
                  the parties and the
                </Text>
              )}
            </ScrollView>
          </View>
        </LinearGradient>
        <Pressable onPress={changeIsBookMark} style={detailStyle.bookmark}>
          {isBookMark ? (
            <Image
              source={require("../../../assets/icons/bookmarkblue_fill.png")}
              style={detailStyle.bookmarkSize}
            />
          ) : (
            <Image
              source={require("../../../assets/icons/bookmarkblue.png")}
              style={detailStyle.bookmarkSize}
            />
          )}
        </Pressable>
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
    minHeight: calculateDynamicWidth(104),
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

import { StyleSheet, Dimensions } from "react-native";
import Colors from "../../constants/colors";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

export const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
  },

  // 모달 뒷배경
  background: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
  },

  // 헤더
  headerContainer: {
    position: "absolute",
    zIndex: 1,
    width: screenWidth,
  },

  // 메인 버튼
  btnContainer: {
    position: "absolute",
    bottom: calculateDynamicWidth(20),
    left: "50%",
    transform: [{ translateX: -calculateDynamicWidth(55) / 2 }],
  },
  addBtn: {
    width: calculateDynamicWidth(55),
    height: calculateDynamicWidth(55),
  },

  // 알림 모달 관련 스타일
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  logo: {
    width: 189,
    height: 35,
  },
  cancelContainer: {
    position: "absolute",
    right: 25,
  },
  cancel: {
    width: 22.92,
    height: 22.92,
  },
  modalEmptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    marginTop: -calculateDynamicWidth(100),
  },
  helpTitle: {
    color: "white",
    fontFamily: "Pretendard-ExtraBold",
    fontSize: calculateDynamicWidth(24),
    textAlign: "center",
  },
  helpText: {
    color: "white",
    fontFamily: "Pretendard-SemiBold",
    fontSize: calculateDynamicWidth(18),
    marginVertical: calculateDynamicWidth(5),
  },
  helpContent: {
    color: "white",
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(18),
    marginVertical: calculateDynamicWidth(5),
  },
  helpContainer: {
    marginTop: calculateDynamicWidth(20),
  },

  // 메모 모달 관련 스타일
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
  memoContent: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
  },

  // 메모 우측 상단 버튼 관련 스타일
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

  // 첨부 사진 스타일
  uploadedImg: {
    width: calculateDynamicWidth(257),
    height: calculateDynamicWidth(106),
    borderRadius: calculateDynamicWidth(10),
    marginTop: calculateDynamicWidth(5),
  },

  // 첨부 사진 상세 스타일
  uploadedFullImg: {
    zIndex: 2,
    borderRadius: calculateDynamicWidth(10),
    position: "absolute",
    top: "50%",
    left: "50%",
  },

  // 첨부 사진 상세 뒷 배경
  uploadedImgBg: {
    flex: 1,
    backgroundColor: "transparent",
    zIndex: 1,
    marginBottom: -screenHeight,
  },

  // 첨부 사진 삭제 아이콘
  bin: {
    width: calculateDynamicWidth(15.61),
    height: calculateDynamicWidth(16),
  },
  binContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    zIndex: 3,
  },

  // 공개 설정 관련 스타일
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

  // 날짜 관련 스타일
  currentDate: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: Colors.hover,
    marginTop: calculateDynamicWidth(4),
  },

  // 태그 관련 스타일
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
  tagText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: Colors.text,
  },
  tagSearchContainer: {
    flexDirection: "row",
    alignItems: "center",
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
  email: {
    fontSize: calculateDynamicWidth(12),
    color: "rgba(44, 44, 44, 0.5)",
  },
  tagResultInnerContainer: {
    marginVertical: calculateDynamicWidth(12),
  },
  closeTagSearch: {
    flex: 1,
    backgroundColor: "transparent",
    zIndex: 1,
    marginTop: -screenHeight,
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
  // RTC View 화면 비율
  video: {
    width: "100%",
    height: "100%",
  },
  // RTC 통신 버튼(Start, Stop) => 향후 삭제 예정
  rtcButton: {
    position: "absolute",
    bottom: 10,
    left: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    width: "90%",
  },
  ObjCircle: {
    position: "absolute",
    width: 100,
    height: 100,
    justifyContent: "center",
    alignItems: "center",
    elevation: 5,
  },
  imageContainer: {
    width: "100%",
    height: "100%",
    position: "relative",
  },
  ObjImg: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
    position: "absolute",
    top: 0,
    left: 0,
  },
  textContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  ObjCircleText: {
    // color: Colors.text,
    color: "white",
    fontSize: 30,
    fontWeight: "bold",
    // opacity: 0.6,
    fontFamily: "Pretendard-SemiBold",
  },
  memoClose: {
    flex: 1,
    backgroundColor: "transparent",
    marginTop: -screenHeight,
  },

  // 물체 학습 시 가운데 포커스 표시

  focusBox: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(60) / 2 },
      { translateX: -calculateDynamicWidth(60) / 2 },
    ],
  },

  focusImg: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
  },
  descriptionBox: {
    flexDirection: "row",
    position: "absolute",
    top: "20%",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  descriptionText: {
    marginLeft: 5,
    fontSize: 20,
    fontWeight: "bold",
    color: "white",
  },
  progressBox: {
    position: "absolute",
    bottom: "20%",
    width: "100%",
    alignItems: "center",
  },
  progressBar: {
    position: "absolute",
    top: 0,
  },
  progressText: {
    width: "100%",
    height: "100%",
    textAlign: "center",
    lineHeight: 25,
    color: "white",
  },
  serviceImage: {
    width: "100%",
    height: "100%",
  },

  // 튜토리얼
  first: {
    width: calculateDynamicWidth(222),
    height: calculateDynamicWidth(65.75),
  },
  firstContainer: {
    position: "absolute",
    top: -calculateDynamicWidth(60),
    left: -calculateDynamicWidth(100),
  },
  second: {
    width: calculateDynamicWidth(218),
    height: calculateDynamicWidth(97),
  },
  secondContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateY: -calculateDynamicWidth(97) / 2 },
      { translateX: -calculateDynamicWidth(218) / 2 },
    ],
  },
  third: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(256),
  },
  thirdContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateX: -calculateDynamicWidth(306) / 2 },
      { translateY: -calculateDynamicWidth(256) / 2 },
    ],
  },
  fourth: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(424),
  },
  fourthContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateX: -calculateDynamicWidth(306) / 2 },
      { translateY: -calculateDynamicWidth(424) / 2 },
    ],
  },
  fifth: {
    width: calculateDynamicWidth(218),
    height: calculateDynamicWidth(74),
  },
  fifthContainer: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: [
      { translateX: -calculateDynamicWidth(218) / 2 },
      { translateY: -calculateDynamicWidth(74) / 2 },
    ],
  },
  startBtnContainer: {
    position: "absolute",
    bottom: 0,
    left: "50%",
    transform: [
      { translateX: -calculateDynamicWidth(270) / 2 },
      { translateY: -calculateDynamicWidth(66) / 2 },
    ],
  },
});

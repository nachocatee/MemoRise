import { StyleSheet, Dimensions } from "react-native";
import Colors from "../../constants/colors";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

export const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
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
    alignItems: "center",
    justifyContent: "center",
    marginTop: -97 / 2,
  },
  modalEmpty: {
    color: "white",
    fontFamily: "Pretendard-SemiBold",
    fontSize: calculateDynamicWidth(18),
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
    borderBlockColor: "#CCCCCC",
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
    color: "#CCCCCC",
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
    marginRight: calculateDynamicWidth(5),
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
});

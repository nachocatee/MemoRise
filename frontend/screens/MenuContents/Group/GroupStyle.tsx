import { StyleSheet, Dimensions } from "react-native";

import Colors from "../../../constants/colors";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";

const screenHeight = Dimensions.get("window").height;
const screenWidth = Dimensions.get("window").width;

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    marginHorizontal: 30,
  },
  titleContainer: {
    marginVertical: calculateDynamicWidth(10),
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: "100%",
  },
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(23),
    color: Colors.text,
    letterSpacing: -0.5,
  },
  emptyContainer: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: screenHeight / 4,
  },
  group: {
    width: calculateDynamicWidth(46.67),
    height: calculateDynamicWidth(42),
  },
  empty: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(20),
    color: "#2c2c2c",
    opacity: 0.6,
    marginTop: calculateDynamicWidth(15),
  },
  findGroupBtn: {
    width: calculateDynamicWidth(113),
    height: calculateDynamicWidth(42),
    backgroundColor: Colors.primary200,
    borderRadius: calculateDynamicWidth(10),
    justifyContent: "center",
    elevation: 4,
    marginTop: calculateDynamicWidth(20),
  },
  btnText: {
    textAlign: "center",
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(20),
    color: Colors.text,
  },
  pressed: {
    backgroundColor: Colors.hover,
  },
  btnContainer: {
    marginBottom: "10%",
  },
  modalContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  contentsContainer: {
    flex: 1,
    alignItems: "center",
    marginTop: calculateDynamicWidth(25),
    justifyContent: "space-between",
  },

  // 그룹 상세 스타일
  settingContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  settingIcon: {
    width: calculateDynamicWidth(19),
    height: calculateDynamicWidth(19),
    marginRight: calculateDynamicWidth(2),
  },
  settingText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: "rgba(44, 44, 44, 0.5)",
  },
  memberWrap: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: "100%",
  },
  memberContainer: {
    width: "100%",
    marginTop: calculateDynamicWidth(30),
  },
  memberImageContainer: {
    elevation: 4,
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(15),
  },
  memberImagebg: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
  },
  memberText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(20),
    color: Colors.text,
    marginLeft: calculateDynamicWidth(18),
  },
  meIcon: {
    width: calculateDynamicWidth(21),
    height: calculateDynamicWidth(21),
  },
  cancelIcon: {
    width: calculateDynamicWidth(11),
    height: calculateDynamicWidth(11),
    marginRight: calculateDynamicWidth(5),
  },
  biggerMeIcon: {
    width: calculateDynamicWidth(45),
    height: calculateDynamicWidth(21),
  },

  // 내 그룹 (not Empty)
  groupContainer: {
    width: "100%",
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "flex-start",
    justifyContent: "space-around",
    paddingVertical: calculateDynamicWidth(30),
  },
  delete: {
    width: calculateDynamicWidth(22),
    height: calculateDynamicWidth(22),
  },
  deleteContainer: {
    position: "absolute",
    top: -calculateDynamicWidth(10),
    left: calculateDynamicWidth(1.5),
    zIndex: 1,
  },

  // 유저 초대하기
  userEmptyContainer: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: screenHeight / 4 - calculateDynamicWidth(79),
  },
  userEmptyIcon: {
    width: calculateDynamicWidth(42),
    height: calculateDynamicWidth(42),
  },
  inviteBtn: {
    width: calculateDynamicWidth(58),
    height: calculateDynamicWidth(27),
    borderRadius: calculateDynamicWidth(10),
    backgroundColor: Colors.primary200,
    elevation: 4,
    justifyContent: "center",
    alignItems: "center",
  },
  inviteBtnText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(16),
    color: Colors.text,
  },
  userListContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginVertical: calculateDynamicWidth(10),
    width: screenWidth - 60,
  },
  // 그룹 찾기
  lockIcon: {
    width: calculateDynamicWidth(30),
    height: calculateDynamicWidth(30),
  },
  lockIconContainer: {
    position: "absolute",
    right: 0,
    top: -calculateDynamicWidth(10),
  },
  emptyResultContainer: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: screenHeight / 4 - calculateDynamicWidth(69),
  },
});

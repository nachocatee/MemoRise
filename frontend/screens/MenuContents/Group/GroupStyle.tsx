import { StyleSheet, Dimensions } from "react-native";

import Colors from "../../../constants/colors";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    marginHorizontal: 30,
  },
  titleContainer: {
    marginTop: calculateDynamicWidth(10),
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: "100%",
  },
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(23),
    color: Colors.text,
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
  contentsContainer: {
    flex: 1,
    alignItems: "center",
    marginTop: calculateDynamicWidth(25),
    justifyContent: "space-between",
  },
  text: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
  },
  line: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.hover,
    height: calculateDynamicWidth(30),
    width: screenWidth - 60,
    marginBottom: calculateDynamicWidth(10),
  },
  itemContainer: {
    marginTop: calculateDynamicWidth(15),
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: screenWidth - 60,
  },
  btnContainer: {
    marginBottom: "10%",
  },
  modalContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },

  // 그룹 상세 스타일
  settingContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: calculateDynamicWidth(25),
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
  },
  memberContainer: {
    width: "100%",
    marginTop: calculateDynamicWidth(30),
  },
  memberInnerContainer: {
    flexDirection: "row",
    alignItems: "center",
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
  memberImage: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(15),
    position: "absolute",
  },
  memberText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(20),
    color: Colors.text,
    marginLeft: calculateDynamicWidth(18),
    // marginBottom: calculateDynamicWidth(8),
  },
  meIcon: {
    width: calculateDynamicWidth(21),
    height: calculateDynamicWidth(21),
  },
});

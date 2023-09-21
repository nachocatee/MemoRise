import { StyleSheet, Dimensions } from "react-native";
import Colors from "../../constants/colors";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

const screenWidth = Dimensions.get("window").width;
const screenHeight = Dimensions.get("window").height;

export const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    backgroundColor: "pink",
  },
  headerContainer: {
    position: "absolute",
    zIndex: 1,
    width: screenWidth,
  },
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
  currentDate: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(14),
    color: "#CCCCCC",
    marginTop: calculateDynamicWidth(4),
  },
  memoInnerContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
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
  memoContent: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    color: Colors.text,
  },
});

import { StyleSheet, Dimensions } from "react-native";

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
  memoBtnContainer: {
    marginVertical: calculateDynamicWidth(5),
    alignItems: "center",
    justifyContent: "center",
  },
  memoBtnText: {
    color: "white",
    fontFamily: "Pretendard-SemiBold",
    fontSize: calculateDynamicWidth(18),
    marginRight: calculateDynamicWidth(10),
    position: "absolute",
    left: screenWidth / 2 - calculateDynamicWidth(150),
  },
  memoBtnWrap: {
    marginTop: screenHeight - calculateDynamicWidth(240),
  },
});

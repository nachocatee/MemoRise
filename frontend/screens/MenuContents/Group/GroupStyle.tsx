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
    // width: calculateDynamicWidth(330),
    marginBottom: calculateDynamicWidth(10),
  },
  itemContainer: {
    marginTop: calculateDynamicWidth(15),
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    width: screenWidth - 60,
    // marginHorizontal: 30,
  },
  btnContainer: {
    marginBottom: "10%",
  },
  modalContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});

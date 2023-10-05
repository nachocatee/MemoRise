import { StyleSheet, Dimensions } from "react-native";

import Colors from "../../../constants/colors";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";

const screenHeight = Dimensions.get("window").height;

export const styles = StyleSheet.create({
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(23),
    color: Colors.text,
    marginLeft: calculateDynamicWidth(30),
    marginTop: calculateDynamicWidth(10),
  },
  container: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: screenHeight / 4,
  },
  icon: {
    width: calculateDynamicWidth(42),
    height: calculateDynamicWidth(42),
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
});

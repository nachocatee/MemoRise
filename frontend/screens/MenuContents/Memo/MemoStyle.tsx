import { StyleSheet } from "react-native";

import Colors from "../../../constants/colors";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";

export const styles = StyleSheet.create({
  title: {
    fontFamily: "Pretendard-Medium",
    fontSize: calculateDynamicWidth(23),
    color: Colors.text,
    marginLeft: 30,
    marginTop: 10,
  },
  container: {
    justifyContent: "center",
    alignItems: "center",
    marginTop: "60%",
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

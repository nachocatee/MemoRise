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
    marginTop: "50%",
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
    marginTop: "7%",
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
});

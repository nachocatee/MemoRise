import { StyleSheet } from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
  },
  text: {
    fontSize: calculateDynamicWidth(18),
    fontFamily: "Pretendard-Medium",
    color: Colors.text,
  },
  emailBox: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#DEDEDE",
    width: calculateDynamicWidth(155),
    borderRadius: calculateDynamicWidth(20) / 2,
    marginTop: calculateDynamicWidth(8),
  },
  email: {
    fontSize: calculateDynamicWidth(15),
    color: Colors.text,
    opacity: 0.5,
  },
  kakao: {
    width: calculateDynamicWidth(12),
    height: calculateDynamicWidth(12),
    marginRight: calculateDynamicWidth(5),
  },
  note: {
    width: calculateDynamicWidth(320),
    height: calculateDynamicWidth(438),
  },
  menuList: {
    position: "absolute",
    top: "18%",
    left: "11%",
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  menuIcon: {
    marginRight: calculateDynamicWidth(20),
    marginVertical: calculateDynamicWidth(15),
  },
  bookmark: {
    width: calculateDynamicWidth(15.52),
    height: calculateDynamicWidth(21),
    marginRight: calculateDynamicWidth(18),
  },
  dotIcon: {
    width: calculateDynamicWidth(12),
    height: calculateDynamicWidth(12),
    borderRadius: calculateDynamicWidth(12) / 2,
    marginLeft: calculateDynamicWidth(2),
  },
  line: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.hover,
    height: calculateDynamicWidth(14),
    width: calculateDynamicWidth(245),
    marginBottom: calculateDynamicWidth(14),
  },
  highlight: {
    width: "120%",
    height: calculateDynamicWidth(10),
    position: "absolute",
    backgroundColor: "#FAEEAE",
    zIndex: -1,
    bottom: 0,
    left: "-10%",
  },
});

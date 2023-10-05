import { StyleSheet } from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "space-between",
    alignItems: "center",
  },
  text: {
    fontSize: calculateDynamicWidth(20),
    fontFamily: "Pretendard-SemiBold",
    color: Colors.text,
    letterSpacing: -0.5,
  },
  profilebox: {
    width: calculateDynamicWidth(103),
    marginTop: "20%",
  },
  userImageContainer: {
    position: "absolute",
    width: calculateDynamicWidth(103),
    height: calculateDynamicWidth(103),
    borderRadius: calculateDynamicWidth(103) / 2,
    elevation: 4,
  },
  userImage: {
    width: calculateDynamicWidth(103),
    height: calculateDynamicWidth(103),
    borderRadius: calculateDynamicWidth(103) / 2,
    position: "absolute",
  },
  album: {
    width: calculateDynamicWidth(31),
    height: calculateDynamicWidth(31),
    position: "absolute",
    bottom: 0,
    right: 0,
  },
  inputBox: {
    width: calculateDynamicWidth(285),
    marginTop: 30,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.text,
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
    marginBottom: 10,
  },
  infoText: {
    fontSize: calculateDynamicWidth(13),
    color: Colors.text,
    opacity: 0.5,
    fontFamily: "Pretendard-Light",
    letterSpacing: -0.5,
  },
  buttonContainer: {
    marginBottom: "10%",
  },
});

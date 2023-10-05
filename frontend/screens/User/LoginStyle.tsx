import { StyleSheet } from "react-native";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

export const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  logo: {
    position: "absolute",
    top: "27%",
    width: calculateDynamicWidth(253),
    height: calculateDynamicWidth(48),
  },
  kakao: {
    position: "absolute",
    bottom: "27%",
  },
  image: {
    width: calculateDynamicWidth(320),
    height: calculateDynamicWidth(333),
  },
  kakaoImage: {
    width: calculateDynamicWidth(230),
    height: calculateDynamicWidth(48),
  },
});

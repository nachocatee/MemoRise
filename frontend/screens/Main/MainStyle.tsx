import { StyleSheet } from "react-native";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

export const styles = StyleSheet.create({
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
});
